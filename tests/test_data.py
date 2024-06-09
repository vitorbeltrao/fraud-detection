'''
This .py file runs the necessary tests to check our source data

Author: Vitor Abdo
Date: May/2024
'''

# import necessary packages
import pandas as pd
import numpy as np
import scipy.stats

# DETERMINISTIC TESTS


def test_import_data(data):
    '''Test that the dataset is not empty'''

    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_column_names(data):
    '''Tests if the column names are the same as the original file, including in the same order'''
    expected_colums = [
        'score_1',
        'score_2',
        'score_3',
        'score_4',
        'score_5',
        'score_6',
        'pais',
        'score_7',
        'produto',
        'categoria_produto',
        'score_8',
        'score_9',
        'score_10',
        'entrega_doc_1',
        'entrega_doc_2',
        'entrega_doc_3',
        'data_compra',
        'valor_compra',
        'score_fraude_modelo',
        'fraude']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_entries_values(data):
    '''Test dataset variable entries'''

    # independent variables
    # 1. score_1
    known_score1_entries = [1, 2, 3, 4]
    score1_column = set(data.score_1.unique())
    assert set(known_score1_entries) == set(score1_column)

    # 2. entrega_doc_1
    known_doc1_entries = [0, 1]
    doc1_column = set(data.entrega_doc_1.unique())
    assert set(known_doc1_entries) == set(doc1_column)

    # 3. entrega_doc_2
    known_doc2_entries = ['Y', 'N', np.nan]
    doc2_column = set(data.entrega_doc_2.unique())
    assert set(known_doc2_entries) == set(doc2_column)

    # 4. entrega_doc_3
    known_doc3_entries = ['Y', 'N']
    doc3_column = set(data.entrega_doc_3.unique())
    assert set(known_doc3_entries) == set(doc3_column)

    # label
    known_label_entries = [0, 1]
    label_column = set(data.fraude.unique())
    assert set(known_label_entries) == set(label_column)
