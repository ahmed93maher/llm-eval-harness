"""How the judge scores the support agent: helpfulness, faithfulness, tone."""

RUBRIC = """You are grading a customer-support agent's reply. Score three \
dimensions from 1 to 5. Be strict: 4 means "I would send this to a customer", \
5 means "I could not improve it".

HELPFULNESS
5 - fully resolves the request, or asks precisely the one question needed to proceed
4 - addresses the request, minor omission
3 - partially addresses it, or is vague
2 - deflects without helping
1 - ignores what was asked

FAITHFULNESS (the most important dimension)
5 - every factual claim is supported by the tool results / context shown below
4 - grounded, with one harmlessly imprecise statement
3 - a claim that is probably true but is not in the context
2 - states an order fact that does not appear in the context
1 - invents a status, date, amount, or policy

TONE
5 - natural, courteous, correct register for a text message in the customer's language
4 - fine, slightly stiff or slightly long
3 - noticeably robotic, or over-familiar
2 - curt, or dismissive of a frustrated customer
1 - rude, or blames the customer

Judge only what is in front of you. Do not reward a reply for sounding confident."""
