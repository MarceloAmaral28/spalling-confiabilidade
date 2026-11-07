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
    def test_named_cases_survive_clearing_fields_and_export_as_values(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()

        def button(label):
            return next(item for item in app.button if item.label == label)

        button('Carregar exemplo').click().run()
        button('Avaliar').click().run()
        app.text_input[0].set_value('=1+1')
        button('Adicionar ao histórico').click().run()
        self.assertTrue(button('Adicionar ao histórico').disabled)
        button('Limpar campos').click().run()
        self.assertEqual(len(app.session_state['evaluation_history']), 1)
        button('Carregar exemplo').click().run()
        app.number_input(key='moisture').set_value(4.0)
        button('Avaliar').click().run()
        with patch.object(st, 'download_button', wraps=st.download_button) as download:
            button('Adicionar ao histórico').click().run()
            self.assertEqual(len(app.exception), 0)
            payload = download.call_args.kwargs['data']

        rows = app.session_state['evaluation_history']
        self.assertEqual([row['Teor de umidade (%)'] for row in rows], [5.0, 4.0])
        self.assertEqual([row['Caso'] for row in rows], ['=1+1', 'Caso 2'])
        with ZipFile(BytesIO(payload)) as workbook:
            self.assertIsNone(workbook.testzip())
            sheet = ElementTree.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
            strings = ElementTree.fromstring(workbook.read('xl/sharedStrings.xml'))
            self.assertEqual(len(sheet.findall('.//x:row', NS)), 3)
            self.assertFalse(sheet.findall('.//x:f', NS))
            self.assertIn('=1+1', [cell.text for cell in strings.findall('.//x:t', NS)])
            self.assertEqual(sheet.find(".//x:c[@r='D3']/x:v", NS).text, '4')
        button('Limpar histórico').click().run()
        self.assertEqual(app.session_state['evaluation_history'], [])
        self.assertIn('evaluation', app.session_state)
