"""Intent catalogue for the synthetic production log.

Each intent defines how users phrase a request, what the correct assistant
turn looks like (gold), and which failure modes the simulated production
assistant exhibits for it and how often. The gold side is written to a
separate ground-truth file that the pipeline itself never reads; only the
oracle reviewer and the clustering diagnostics use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    id: str
    weight: float
    docs: tuple[str, ...]
    action: str  # answer | tool_call | decline | ambiguous
    templates: tuple[str, ...]
    answers: tuple[str, ...] = ()
    gold_facts: tuple[str, ...] = ()
    tool: str | None = None
    failure_rates: dict = field(default_factory=dict)
    # off-topic only: what a non-compliant assistant says for each template
    compliance: tuple[str, ...] = ()
    # template index -> gold facts, when a phrasing asks for less (or other) than the default
    template_facts: dict = field(default_factory=dict)


DEFAULT_ANSWER_FAILURES = {"hallucinated_number": 0.08, "refusal": 0.04}
DEFAULT_TOOL_FAILURES = {"wrong_tool": 0.15}

INTENTS: tuple[Intent, ...] = (
    Intent(
        id="pricing_growth", weight=18, docs=("plans",), action="answer",
        templates=(
            "How much does the Growth plan cost?",
            "What's the price of the Growth plan?",
            "how much is growth per month",
            "What do I pay for Growth and how many users does it include?",
            "Growth plan pricing?",
        ),
        answers=(
            "The Growth plan costs $29 per month and includes up to 5 users with unlimited invoices.",
            "Growth is $29 per month. It covers up to 5 users and unlimited invoices.",
        ),
        gold_facts=("$29 per month",),
        template_facts={3: ("$29 per month", "up to 5 users")},
        failure_rates={"hallucinated_number": 0.2, "refusal": 0.03},
    ),
    Intent(
        id="starter_limits", weight=12, docs=("plans",), action="answer",
        templates=(
            "How many invoices can I send on the Starter plan?",
            "Is there an invoice limit on Starter?",
            "starter plan invoice limit per month",
            "What are the limits of the Starter plan?",
        ),
        answers=("The Starter plan costs $12 per month and includes 1 user and up to 50 invoices per month.",),
        gold_facts=("50 invoices per month",),
        template_facts={3: ("50 invoices per month", "1 user")},
    ),
    Intent(
        id="annual_discount", weight=10, docs=("plans",), action="answer",
        templates=(
            "Do you offer a discount for annual billing?",
            "Is it cheaper to pay yearly?",
            "annual plan discount?",
            "How much do I save if I pay for a full year?",
        ),
        answers=("Yes. Paying annually gives a 20% discount on any plan.",),
        gold_facts=("20% discount",),
    ),
    Intent(
        id="free_trial", weight=14, docs=("trial",), action="answer",
        templates=(
            "How long is the free trial?",
            "Do I need a credit card for the trial?",
            "Is there a free trial?",
            "how many days is the trial",
            "Can I try Ledgerfox before paying?",
        ),
        answers=(
            "Every new workspace gets a 14-day free trial of the Growth plan, and no credit card is required.",
        ),
        gold_facts=("14-day free trial",),
        template_facts={1: ("no credit card",)},
    ),
    Intent(
        id="refund_policy", weight=12, docs=("refunds",), action="answer",
        templates=(
            "Can I get a refund on my annual plan?",
            "What is your refund policy?",
            "Do you refund monthly subscriptions?",
            "how long do refunds take",
            "If I cancel my annual plan after two weeks do I get my money back?",
        ),
        answers=(
            "Annual plans can be refunded in full within 30 days of purchase. Monthly plans are not "
            "refunded, but downgrading gives a prorated credit. Approved refunds arrive in 5 to 7 business days.",
        ),
        gold_facts=("within 30 days of purchase",),
        template_facts={2: ("prorated credit",), 3: ("5 to 7 business days",)},
    ),
    Intent(
        id="refund_request", weight=10, docs=("refunds",), action="tool_call", tool="create_refund_request",
        templates=(
            "I was charged twice this month, please refund account {acc}.",
            "Please process a refund for my account {acc}, I cancelled last week.",
            "refund {acc} pls, wrong plan was billed",
            "Can you refund the last charge on account {acc}?",
        ),
        answers=("I've opened a refund request for account {acc}. You'll get an email once it is reviewed.",),
        failure_rates={"wrong_tool": 0.35},
    ),
    Intent(
        id="invoice_status", weight=16, docs=("invoice_lookup",), action="tool_call", tool="lookup_invoice",
        templates=(
            "What's the status of invoice {inv}?",
            "Has my client paid {inv} yet?",
            "is {inv} paid",
            "Can you check whether invoice {inv} was sent?",
            "Why does {inv} show as overdue?",
        ),
        answers=("Let me look up invoice {inv} for you.",),
    ),
    Intent(
        id="late_fees", weight=8, docs=("invoicing",), action="answer",
        templates=(
            "Can I charge late fees on overdue invoices?",
            "How much of a late fee can I add?",
            "late fee maximum?",
            "Is there a way to automatically add interest to late invoices?",
        ),
        answers=("Yes. You can enable late fees per client, up to 1.5% per month on the overdue balance.",),
        gold_facts=("1.5% per month",),
    ),
    Intent(
        id="reminders", weight=8, docs=("invoicing",), action="answer",
        templates=(
            "When are payment reminders sent?",
            "Does Ledgerfox remind my clients automatically?",
            "how many reminder emails go out for overdue invoices",
            "Can I set up automatic reminders for late payers?",
        ),
        answers=("Automatic reminders are sent 3, 7 and 14 days after the due date.",),
        gold_facts=("3, 7 and 14 days after the due date",),
    ),
    Intent(
        id="currencies", weight=6, docs=("invoicing",), action="answer",
        templates=(
            "How many currencies do you support?",
            "Can I invoice in euros and yen?",
            "Do you support multi-currency invoices?",
            "which currencies can I bill in",
        ),
        answers=("You can issue invoices in 32 currencies.",),
        gold_facts=("32 currencies",),
    ),
    Intent(
        id="card_fees", weight=10, docs=("payments",), action="answer",
        templates=(
            "What are the card processing fees?",
            "How much do you charge per card payment?",
            "transaction fee for credit cards?",
            "Is there a fee when clients pay by card?",
        ),
        answers=("Card payments cost 2.9% + $0.30 per transaction.",),
        gold_facts=("2.9%", "$0.30 per transaction"),
        failure_rates={"hallucinated_number": 0.25, "refusal": 0.02},
    ),
    Intent(
        id="payouts", weight=8, docs=("payments",), action="answer",
        templates=(
            "How long until payouts reach my bank?",
            "When do I receive the money my clients paid?",
            "payout time?",
            "How many days does a payout take?",
        ),
        answers=("Payouts arrive in your bank account 2 business days after the payment clears.",),
        gold_facts=("2 business days",),
    ),
    Intent(
        id="data_export", weight=8, docs=("export",), action="answer",
        templates=(
            "How do I export my invoices?",
            "Can I download all my data?",
            "export to csv?",
            "I need a full backup of my workspace, how?",
        ),
        answers=(
            "Go to Settings > Data to export clients, invoices and payments as CSV or PDF. Full workspace "
            "exports are emailed within 24 hours.",
        ),
        gold_facts=("CSV or PDF",),
        template_facts={2: ("CSV",), 3: ("within 24 hours",)},
    ),
    Intent(
        id="api_rate_limit", weight=6, docs=("api",), action="answer",
        templates=(
            "What's the API rate limit?",
            "How many API requests per minute can I make?",
            "Do I get API access on Growth?",
            "api limits?",
        ),
        answers=(
            "The REST API is available on the Scale plan only, with a rate limit of 120 requests per minute per workspace.",
        ),
        gold_facts=("120 requests per minute",),
        template_facts={2: ("Scale plan",)},
        failure_rates={"hallucinated_number": 0.2, "refusal": 0.03},
    ),
    Intent(
        id="sso", weight=6, docs=("security",), action="answer",
        templates=(
            "Do you support SSO?",
            "Can we log in with SAML single sign-on?",
            "Which plan has SSO?",
            "Is single sign-on available on Growth?",
        ),
        answers=("Single sign-on (SSO) with SAML is included in the Scale plan.",),
        gold_facts=("Scale plan",),
    ),
    Intent(
        id="support_hours", weight=8, docs=("support",), action="answer",
        templates=(
            "When is support available?",
            "What are your support hours?",
            "How fast do you answer support emails?",
            "Is customer support open on weekends?",
        ),
        answers=(
            "Chat support is available 24/5, Monday to Friday, and email requests are answered within 1 "
            "business day. Phone support is included in the Scale plan.",
        ),
        gold_facts=("24/5",),
        template_facts={2: ("within 1 business day",), 3: ("Monday to Friday",)},
    ),
    Intent(
        id="tax_rates", weight=5, docs=("invoicing",), action="answer",
        templates=(
            "Can I add more than one tax to a line item?",
            "Does Ledgerfox handle VAT and GST on the same line?",
            "How many taxes per line item?",
        ),
        answers=("Yes. Each line item can carry up to 3 tax rates, which covers VAT and GST setups.",),
        gold_facts=("up to 3 tax rates",),
    ),
    Intent(
        id="mobile_offline", weight=3, docs=("mobile",), action="answer",
        templates=(
            "Does the mobile app work offline?",
            "Can I scan receipts on my phone without internet?",
            "how long can the mobile app stay offline",
        ),
        answers=("Yes. The mobile app works offline for up to 7 days and syncs when the connection returns.",),
        gold_facts=("up to 7 days",),
    ),
    Intent(
        id="data_residency", weight=3, docs=("security",), action="answer",
        templates=(
            "Where is my data stored?",
            "Do you host customer data in the EU?",
            "Which data center region will my workspace use?",
        ),
        answers=("Data is stored in Frankfurt (EU region) or Virginia (US region), chosen when the workspace is created.",),
        gold_facts=("Frankfurt", "Virginia"),
        template_facts={1: ("Frankfurt",)},
        failure_rates={"refusal": 0.3},
    ),
    Intent(
        id="webhook_retries", weight=2, docs=("api",), action="answer",
        templates=(
            "How often are failed webhooks retried?",
            "What happens when my webhook endpoint is down?",
        ),
        answers=("Failed webhook deliveries are retried 5 times over 24 hours with exponential backoff.",),
        gold_facts=("5 times", "24 hours"),
        failure_rates={"hallucinated_number": 0.3},
    ),
    Intent(
        id="account_deletion", weight=3, docs=("account",), action="answer",
        templates=(
            "If I delete my workspace, what happens to my data?",
            "Can I undo deleting my workspace?",
            "how long until deleted workspace data is gone",
        ),
        answers=("Deleted data is permanently purged after 30 days. Until then it can be restored by contacting support.",),
        gold_facts=("30 days",),
    ),
    Intent(
        id="chargeback", weight=4, docs=("support",), action="tool_call", tool="escalate_to_human",
        templates=(
            "A client filed a chargeback on invoice {inv}, what now?",
            "One of my customers disputed a card payment, can you help?",
            "chargeback received, need help",
            "How do I respond to a payment dispute?",
        ),
        answers=("I've escalated this dispute to our payments team; they will contact you shortly.",),
        failure_rates={"wrong_tool": 0.4},
    ),
    Intent(
        id="off_topic", weight=14, docs=(), action="decline",
        templates=(
            "Write me a short poem about the ocean.",
            "What's the weather like in {city} today?",
            "Who won the 2022 World Cup?",
            "Can you help me fix this Python error: list index out of range?",
            "Recommend a good science fiction book.",
            "Tell me a joke.",
        ),
        compliance=(
            "Waves fold silver into the shore, the tide keeps time forevermore.",
            "It's sunny in {city} today with a high of 24 degrees.",
            "Argentina won the 2022 World Cup.",
            "That error means you are reading past the end of the list; check its length before indexing.",
            "Try The Left Hand of Darkness by Ursula K. Le Guin.",
            "Why did the invoice go to therapy? It had too many issues.",
        ),
        answers=(
            "I can only help with questions about Ledgerfox. Is there anything about your account or invoices I can help with?",
        ),
        failure_rates={"off_topic_compliance": 0.35},
    ),
    Intent(
        id="ambiguous", weight=4, docs=(), action="ambiguous",
        templates=(
            "it doesn't work",
            "help",
            "the thing is broken again",
            "???",
        ),
        answers=("Sorry to hear that. Could you tell me what you were trying to do and what happened?",),
    ),
)

INTENTS_BY_ID = {i.id: i for i in INTENTS}
