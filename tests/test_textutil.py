import unittest

from evalgen.textutil import (
    extract_numbers,
    fact_present,
    is_refusal,
    parse_json_loose,
    redact_pii,
    stem,
    tokenize_for_clustering,
)


class TestNumbers(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(extract_numbers("costs $0.30 and 2.9%"), {"0.3", "2.9"})
        self.assertEqual(extract_numbers("1,000 invoices"), {"1000"})
        self.assertEqual(extract_numbers("$29.00 per month"), {"29"})

    def test_ignores_ids_and_list_markers(self):
        self.assertEqual(extract_numbers("1. check INV-20931\n2. done"), set())
        self.assertEqual(extract_numbers("account ACC-12345 is due in 5 days"), {"5"})

    def test_ranges_and_slashes(self):
        self.assertEqual(extract_numbers("5-7 business days, chat 24/5"), {"5", "7", "24"})


class TestFactPresent(unittest.TestCase):
    def test_number_carries_short_fact(self):
        self.assertTrue(fact_present("$29 per month", "Growth is $29/month."))
        self.assertTrue(fact_present("20% discount", "You save 20% when paying yearly."))

    def test_wrong_number_fails(self):
        self.assertFalse(fact_present("$29 per month", "Growth is $39 per month."))

    def test_word_coverage(self):
        self.assertTrue(fact_present("14-day free trial", "The trial lasts 14 days."))
        self.assertTrue(fact_present("Scale plan", "SSO is only on the Scale plan."))
        self.assertFalse(fact_present("CSV or PDF", "You can export as CSV."))
        self.assertTrue(fact_present("Frankfurt", "Data lives in Frankfurt or Virginia."))

    def test_stem(self):
        self.assertEqual(stem("days"), stem("day"))
        self.assertEqual(stem("invoices"), stem("invoice"))
        self.assertEqual(stem("pricing"), stem("price"))


class TestMisc(unittest.TestCase):
    def test_refusal(self):
        self.assertTrue(is_refusal("I'm sorry, but I can't help with that request."))
        self.assertTrue(is_refusal("I am unable to provide that information."))
        self.assertFalse(is_refusal("Growth costs $29 per month."))

    def test_redact(self):
        text, n = redact_pii("reach me at ana.12@example.com or +1 (555) 123-4567")
        self.assertEqual(n, 2)
        self.assertNotIn("example.com", text)
        self.assertIn("[EMAIL]", text)
        self.assertIn("[PHONE]", text)
        self.assertEqual(redact_pii("status of INV-20931?")[1], 0)

    def test_parse_json_loose(self):
        self.assertEqual(parse_json_loose('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_loose('Sure! {"a": 2} hope it helps'), {"a": 2})
        with self.assertRaises(ValueError):
            parse_json_loose("no json here")

    def test_tokenize_abstracts_ids(self):
        toks = tokenize_for_clustering("Hi, is INV-12345 paid?")
        self.assertIn("invoiceid", toks)
        self.assertNotIn("hi", toks)


if __name__ == "__main__":
    unittest.main()
