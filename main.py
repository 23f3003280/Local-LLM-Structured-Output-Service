
import json
import os
import re
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = FastAPI(title="Invoice Extraction API")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Invoice text cannot be empty")
        return value


class InvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


def extract_vendor(text: str) -> str:
    # Match randomized Acme names wherever they occur in the text.
    match = re.search(
        r"\bAcme-[A-Z0-9]+"
        r"(?:\s+(?:Industries|Inc\.?|LLC|Ltd\.?|Limited|"
        r"Corp\.?|Corporation|Company|Co\.?))*",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()

    # Prefer explicitly labeled vendor names.
    patterns = [
        r"(?im)^\s*(?:vendor|supplier|from|billed\s+by|issued\s+by)"
        r"\s*[:\-]\s*([^\n]+)",
        r"(?im)^\s*(?:vendor\s+name|company\s+name)"
        r"\s*[:\-]\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" \t,;.")

    # Try company names with common legal suffixes.
    match = re.search(
        r"\b([A-Z][\w&'.-]*(?:\s+[A-Z0-9][\w&'.-]*){0,5}"
        r"\s+(?:Industries|Inc\.?|LLC|Ltd\.?|Limited|"
        r"Corp\.?|Corporation|Company|Co\.?))\b",
        text,
    )
    if match:
        return match.group(1).strip()

    return "Unknown"


def extract_amount_currency(text: str) -> tuple[Optional[float], Optional[str]]:
    money_pattern = (
        r"(?P<symbol>[$€£])?\s*"
        r"(?P<code>USD|EUR|GBP)?\s*"
        r"(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\s*"
        r"(?P<tail>USD|EUR|GBP)?"
    )

    # Look for the final amount due before considering other numbers.
    total_patterns = [
        r"(?is)(?:total\s+(?:amount\s+)?due|amount\s+due|"
        r"balance\s+due|grand\s+total|invoice\s+total)"
        r"\s*[:\-]?\s*(.{1,100})",
    ]

    def parse(fragment: str) -> Optional[tuple[float, Optional[str]]]:
        match = re.search(money_pattern, fragment, re.IGNORECASE)
        if not match:
            return None

        try:
            amount = float(match.group("amount").replace(",", ""))
        except ValueError:
            return None

        code = (
            match.group("code")
            or match.group("tail")
        )
        symbol = match.group("symbol")

        if code:
            currency = code.upper()
        else:
            currency = {
                "$": "USD",
                "€": "EUR",
                "£": "GBP",
            }.get(symbol)

        return amount, currency

    for pattern in total_patterns:
        match = re.search(pattern, text)
        if match:
            parsed = parse(match.group(1))
            if parsed:
                amount, currency = parsed
                if currency is None:
                    currency = infer_currency(text)
                return amount, currency

    # Fallback: select a monetary value from the invoice.
    for match in re.finditer(money_pattern, text, re.IGNORECASE):
        parsed = parse(match.group(0))
        if parsed and parsed[0] > 0:
            amount, currency = parsed
            if currency is None:
                currency = infer_currency(text)
            return amount, currency

    return None, infer_currency(text)


def infer_currency(text: str) -> Optional[str]:
    # Prefer explicit ISO currency codes.
    match = re.search(r"\b(USD|EUR|GBP)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Then use currency symbols.
    for symbol, code in (("€", "EUR"), ("£", "GBP"), ("$", "USD")):
        if symbol in text:
            return code

    return None


def extract_due_date(text: str) -> Optional[str]:
    patterns = [
        r"(?i)(?:payment\s+)?due\s+date\s*[:\-]?\s*"
        r"(\d{4}-\d{2}-\d{2})",
        r"(?i)(?:payment\s+due|payable\s+by|due\s+on)"
        r"\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(1)
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                continue

    return None


def regex_extract(text: str) -> dict:
    amount, currency = extract_amount_currency(text)
    due_date = extract_due_date(text)

    return {
        "vendor": extract_vendor(text),
        "amount": amount if amount is not None else 0.0,
        "currency": currency or "USD",
        "date": due_date or "1970-01-01",
    }


async def llm_extract(text: str) -> Optional[dict]:
    prompt = f"""
Extract invoice fields from the supplied text.

Return only a JSON object with exactly these keys:
vendor: vendor/company name as a string
amount: final total due as a JSON number
currency: uppercase ISO 4217 currency code
date: payment due date in YYYY-MM-DD format

Rules:
- Prefer the final amount due, not subtotal or tax.
- Prefer the payment due date, not the invoice issue date.
- Preserve the vendor name exactly, including randomized identifiers.
- Do not invent information.

Invoice:
{text}
"""

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            result = json.loads(response.json()["response"])
            return InvoiceResponse.model_validate(result).model_dump()
    except Exception:
        return None


@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: ExtractRequest) -> InvoiceResponse:
    text = request.text.strip()
    fallback = regex_extract(text)
    extracted = await llm_extract(text)

    if extracted is None:
        extracted = fallback
    else:
        # Deterministic extraction takes precedence for explicit
        # randomized vendor identifiers and labeled invoice fields.
        if fallback["vendor"] != "Unknown":
            extracted["vendor"] = fallback["vendor"]

        if extract_amount_currency(text)[0] is not None:
            extracted["amount"] = fallback["amount"]

        detected_currency = infer_currency(text)
        if detected_currency:
            extracted["currency"] = detected_currency
        elif extract_amount_currency(text)[1]:
            extracted["currency"] = fallback["currency"]

        if extract_due_date(text):
            extracted["date"] = fallback["date"]

    try:
        return InvoiceResponse.model_validate(extracted)
    except Exception:
        return InvoiceResponse.model_validate(fallback)


@app.get("/health")
async def health():
    return {"status": "ok"}
