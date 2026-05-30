"""
Pydantic schema for Medical / Clinical document extraction.

Covers: patient records, clinical notes, discharge summaries,
lab reports, prescriptions, referral letters.
"""

from typing import Optional
from pydantic import BaseModel, Field
from .base import ExtractionResultBase


class Medication(BaseModel):
    """A single medication entry."""

    name: str = Field(description="Drug name, generic or brand.")
    dosage: Optional[str] = Field(
        default=None, description="Dose amount and unit, e.g. '500 mg', '10 mcg'."
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Administration schedule, e.g. 'twice daily', 'QID', 'PRN'.",
    )


class MedicalExtractionResult(ExtractionResultBase):
    """Structured extraction output for medical / clinical documents."""

    document_type: str = "medical"

    patient_name: Optional[str] = Field(
        default=None,
        description="Full name of the patient. Look for 'Patient:', 'Name:'.",
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description="Patient's date of birth. Look for 'DOB:', 'Date of Birth:'. Preserve the format found in the document.",
    )
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="ICD-10 or ICD-9 codes, e.g. ['E11.9', 'I10'].",
    )
    diagnosis_descriptions: list[str] = Field(
        default_factory=list,
        description="Plain-text diagnosis descriptions matching the codes above.",
    )
    medications: list[Medication] = Field(
        default_factory=list,
        description="All medications listed in the document.",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Treating physician or clinician. Look for 'Physician:', 'Provider:', 'Signed by:'.",
    )
    visit_date: Optional[str] = Field(
        default=None,
        description="Date of visit or service. Look for 'Date of Service:', 'Visit Date:', 'Encounter Date:'.",
    )
    facility: Optional[str] = Field(
        default=None,
        description="Hospital, clinic, or facility name. Often appears in the document header.",
    )
