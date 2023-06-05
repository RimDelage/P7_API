###      Test unitaire
import unittest
from api import *


class TestPrediction(unittest.TestCase):

    def test_idClient_Existance(self):
         id_client = 100004
         self.assertIn(id_client,list_id_clients(),"The client ID does not exist.")

    def test_predict_client(self):
        id_client = 100004  # ID client de test
        expected_prediction = 0.25569589095556944  # Prédiction attendue pour l'ID client de test

        # Appel de la fonction predict_client
        prediction = predict_client(id_client)

        # Vérification de la prédiction
        self.assertEqual(prediction, expected_prediction, "The prediction does not match the expected value.")

if __name__ == '__main__':
    unittest.main()
