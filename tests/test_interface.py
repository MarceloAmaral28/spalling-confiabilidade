"""Fluxo de formulário, conversão de umidade e atualização da avaliação."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class InterfaceTests(unittest.TestCase):
    def test_form_flow(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.number_input), 8)
        self.assertTrue(all(field.value is None for field in app.number_input))
        app.button[2].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Preencha:' in error.value for error in app.error))
        self.assertNotIn('evaluation', app.session_state)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(all(field.value is not None for field in app.number_input))
        app.button[2].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)
        result = app.session_state['evaluation']
        self.assertAlmostEqual(result.inputs['Moisture content'], app.number_input(key='moisture').value / 100)
        app.number_input(key='temperature').set_value(1250.0)
        app.button[2].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Temperatura máxima de exposição', app.session_state['evaluation'].outside_ranges)
        app.number_input(key='water_binder').set_value(0.0)
        app.button[2].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('evaluation', app.session_state)
        app.button[1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(all(field.value is None for field in app.number_input))
        self.assertNotIn('evaluation', app.session_state)


if __name__ == '__main__':
    unittest.main()
