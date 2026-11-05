"""Facts worked out by plain Python: dates and age. The LLM never does date arithmetic.

These facts are passed to the LLM as plain sentences. Which policy rule applies to the claim is left to the
LLM, which reads the retrieved clauses; this file only computes the numbers.
"""
import calendar
from dataclasses import dataclass
from datetime import date, timedelta

from claims_rag.schemas import Claim


def add_months(start: date, months: int) -> date:
    """Return the date that is `months` calendar months after `start`.

    If the target month is too short, the day is cut back: 31 January plus one month is 28 (or 29) February.
    """
    # Count months from year 0 with January as month 0, add the months, then convert back to year and month.
    month_count = start.year * 12 + (start.month - 1) + months
    year = month_count // 12
    month = month_count % 12 + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(start.day, last_day_of_month))


@dataclass
class Facts:
    plan: str
    days_of_cover: int  # days from the policy start date to the treatment date
    months_of_cover: int  # whole months completed
    extra_days: int  # days left over after those whole months
    end_30_days: date  # the date the 30-day waiting period ends
    passed_30_days: bool
    end_24_months: date
    passed_24_months: bool
    end_36_months: date
    passed_36_months: bool
    age: int
    age_60_or_more: bool

    def as_statements(self) -> list[str]:
        """Write the facts as plain sentences for the prompt."""
        return [
            f"Plan: {self.plan}.",
            f"Cover at the treatment date: {self.months_of_cover} months and {self.extra_days} days "
            f"({self.days_of_cover} days in total).",
            describe_mark("30-day", self.end_30_days, self.passed_30_days),
            describe_mark("24-month", self.end_24_months, self.passed_24_months),
            describe_mark("36-month", self.end_36_months, self.passed_36_months),
            describe_age(self.age, self.age_60_or_more),
        ]


def describe_mark(label: str, end_date: date, passed: bool) -> str:
    """One sentence saying whether a waiting-period mark has been passed at the treatment date."""
    if passed:
        status = "HAS been passed"
    else:
        status = "has NOT been passed"
    return f"The {label} mark (ends {end_date.isoformat()}) {status} at the treatment date."


def describe_age(age: int, is_60_or_more: bool) -> str:
    """One sentence saying whether the claimant is old enough for the age-based co-payment."""
    if is_60_or_more:
        verb = "is"
    else:
        verb = "is not"
    return f"The claimant {verb} aged 60 or more at the treatment date (age given as {age})."


def compute_facts(claim: Claim) -> Facts:
    """Work out the waiting-period marks and the age flag for one claim.

    The two conventions below follow the wording of the policy:
    - Clause 8.1 excludes treatment that begins "within 30 days after" the start, so the 30-day mark counts
      as passed only once the days of cover are MORE than 30.
    - Clauses 8.3 and 9.1 need months of cover "completed", so those marks count as passed ON the
      anniversary date itself.
    """
    policy_start = claim.policy_start_date
    treatment_date = claim.treatment_date
    if treatment_date < policy_start:
        raise ValueError(f"{claim.claim_id}: treatment_date is before policy_start_date")

    # Whole months of cover. First count the calendar months between the two dates, then step back one
    # month if the last month is not complete yet (the treatment day is earlier in the month than the start day).
    whole_months = (treatment_date.year - policy_start.year) * 12 + treatment_date.month - policy_start.month
    if add_months(policy_start, whole_months) > treatment_date:
        whole_months -= 1
    extra_days = (treatment_date - add_months(policy_start, whole_months)).days
    total_days = (treatment_date - policy_start).days

    # The date each waiting period ends.
    end_30_days = policy_start + timedelta(days=30)
    end_24_months = add_months(policy_start, 24)
    end_36_months = add_months(policy_start, 36)

    return Facts(
        plan=claim.plan,
        days_of_cover=total_days,
        months_of_cover=whole_months,
        extra_days=extra_days,
        end_30_days=end_30_days,
        passed_30_days=total_days > 30,
        end_24_months=end_24_months,
        passed_24_months=treatment_date >= end_24_months,
        end_36_months=end_36_months,
        passed_36_months=treatment_date >= end_36_months,
        age=claim.age,
        age_60_or_more=claim.age >= 60,
    )
