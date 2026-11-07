"""Verifica o histórico da sessão e o conteúdo do Excel exportado."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


class HistoryTests(unittest.TestCase):
    def test_only_valid_submissions_are_numbered_and_exported(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()

        def button(label):
            return next(item for item in app.button if item.label == label)

        self.assertTrue(app.get('download_button')[0].proto.disabled)
        button('Avaliar').click().run()
        self.assertEqual(app.session_state['evaluation_history'], [])
        button('Carregar exemplo').click().run()
        button('Avaliar').click().run()
        self.assertEqual(len(app.session_state['evaluation_history']), 1)
        app.run()
        self.assertEqual(len(app.session_state['evaluation_history']), 1)
        button('Limpar campos').click().run()
        self.assertEqual(len(app.session_state['evaluation_history']), 1)
        button('Carregar exemplo').click().run()
        app.number_input(key='water_binder').set_value(0.0)
        button('Avaliar').click().run()
        self.assertTrue(app.error)
        self.assertEqual(len(app.session_state['evaluation_history']), 1)
        button('Carregar exemplo').click().run()
        app.number_input(key='moisture').set_value(4.0)
        with patch.object(st, 'download_button', wraps=st.download_button) as download:
            button('Avaliar').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(download.call_args.kwargs['disabled'])
            payload = download.call_args.kwargs['data']

        rows = app.session_state['evaluation_history']
        self.assertEqual([row['Teor de umidade (%)'] for row in rows], [5.0, 4.0])
        self.assertEqual([row['Caso'] for row in rows], ['Caso 1', 'Caso 2'])
        with ZipFile(BytesIO(payload)) as workbook:
            self.assertIsNone(workbook.testzip())
            sheet = ElementTree.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
            self.assertEqual(len(sheet.findall('.//x:row', NS)), 3)
            self.assertEqual(sheet.find(".//x:c[@r='D3']/x:v", NS).text, '4')
        fresh = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertEqual(fresh.session_state['evaluation_history'], [])
