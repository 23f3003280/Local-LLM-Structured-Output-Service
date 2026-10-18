
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
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Invoice text cannot be blank")
        return value


class InvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        datetime.strptime(value, "%Y-%m-%d")
        return value


def regex_extract(text: str) -> dict:
    """Extract fields directly from the invoice as a fallback."""
    result = {
        "vendor": "Unknown",
        "amount": 0.0,
        "currency": "USD",
        "date": "1970-01-01",
    }

    if not text or not text.strip():
        return result

    # Prefer the company name on a vendor/supplier line.
    vendor_patterns = [
        r"(?im)^\s*(?:vendor|supplier|from|billed\s+by|issued\s+by)"
        r"\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*((?:[A-Z][\w&'.-]*\s+){0,6}"
        r"(?:Industries|Inc\.?|LLC|Ltd\.?|Limited|Corp\.?|"
        r"Corporation|Company|Co\.?))\b[^\n]*$",
    ]

    for pattern in vendor_patterns:
        match = re.search(pattern, text)
        if match:
            result["vendor"] = match.group(1).strip(" \t,;")
            break

    # Prefer an explicit total due / amount due over subtotals.
    money_pattern = (
        r"(?P<symbol>[$€£])?\s*"
        r"(?P<code>USD|EUR|GBP)?\s*"
        r"(?P<number>\d[\d,]*(?:\.\d{1,2})?)\s*"
        r"(?P<tail>USD|EUR|GBP)?"
    )

    total_patterns = [
        r"(?im)(?:total\s+(?:amount\s+)?due|amount\s+due|"
        r"balance\s+due|grand\s+total|invoice\s+total)"
        r"\s*[:\-]?\s*(.{1,60})",
    ]

    def parse_money(fragment: str) -> Optional[tuple]:
        m = re.search(money_pattern, fragment, re.I)
        if not m:
            return None
        try:
            amount = float(m.group("number").replace(",", ""))
        except ValueError:
            return None

        code = (m.group("code") or m.group("tail") or "").upper()
        symbol = m.group("symbol")
        if not code:
            code = {"$": "USD", "€": "EUR", "£": "GBP"}.get(
                symbol, ""
            )
        return amount, code

    for pattern in total_patterns:
        match = re.search(pattern, text)
        if match:
            parsed = parse_money(match.group(1))
            if parsed:
                result["amount"] = parsed[0]
                if parsed[1]:
                    result["currency"] = parsed[1]
                break

    # If no labeled total was found, inspect monetary values.
    if result["amount"] == 0:
        matches = list(re.finditer(money_pattern, text, re.I))
        for match in matches:
            parsed = parse_money(match.group(0))
            if parsed and parsed[0] > 0:
                result["amount"] = parsed[0]
                if parsed[1]:
                    result["currency"] = parsed[1]
                break

    # Infer currency from the invoice even if the total has no code.
    if result["currency"] == "USD":
        currency_match = re.search(
            r"\b(USD|EUR|GBP)\b|([$€£])", text, re.I
        )
        if currency_match:
            code = currency_match.group(1)
            symbol = currency_match.group(2)
            result["currency"] = (
                code.upper() if code else
                {"$": "USD", "€": "EUR", "£": "GBP"}[symbol]
            )

    # Prefer due/payment dates, then other ISO dates.
    date_patterns = [
        r"(?i)(?:payment\s+)?due\s+date\s*[:\-]?\s*"
        r"(\d{4}-\d{2}-\d{2})",
        r"(?i)(?:payable\s+by|payment\s+due|due\s+on)"
        r"\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                result["date"] = candidate
                break
            except ValueError:
                continue

    return result


async def llm_extract(text: str) -> Optional[dict]:
    """Ask a local Ollama model to interpret the invoice."""
    prompt = f"""
Extract the following fields from this invoice text:
- vendor: the exact vendor/company name
- amount: the final total due, as a JSON number
- currency: ISO 4217 three-letter uppercase currency code
- date: the payment due date in YYYY-MM-DD format

Use the explicitly stated amount due and due date, not the
subtotal, invoice issue date, or payment date.
Do not invent values. Return only a JSON object with exactly
these four keys: vendor, amount, currency, date.

Invoice text:
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
            payload = response.json()
            extracted = json.loads(payload["response"])

            # Validate before allowing the result to be used.
            return InvoiceResponse.model_validate(extracted).model_dump()
    except Exception:
        return None


@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: ExtractRequest) -> InvoiceResponse:
    text = request.text.strip()

    # Regex provides a useful baseline even when the LLM is unavailable.
    fallback = regex_extract(text)

    # The LLM interprets free-form wording and vendor names.
    extracted = await llm_extract(text)

    if extracted is not None:
        # Prefer LLM interpretation, while using explicit regex
        # results when the model omits or corrupts a field.
        for key in ("vendor", "amount", "currency", "date"):
            value = extracted.get(key)
            if value is None or value == "":
                extracted[key] = fallback[key]

        # Use explicitly labeled dates and totals when detected.
        if fallback["date"] != "1970-01-01":
            extracted["date"] = fallback["date"]
        if fallback["amount"] != 0:
            extracted["amount"] = fallback["amount"]
        if fallback["currency"]:
            extracted["currency"] = fallback["currency"]

        try:
            return InvoiceResponse.model_validate(extracted)
        except Exception:
            pass

    return InvoiceResponse.model_validate(fallback)


@app.get("/health")
async def health():
    return {"status": "ok"}
