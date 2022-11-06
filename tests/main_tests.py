import time

from src.train import train_pipeline

from unittest import TestCase


class TestLRUCache(TestCase):
    def test_model(self):
        ts = time.time()
        self.train_config_path = "configs/default_train.yaml"

        x_test, y_test = train_pipeline(self.train_config_path, ts)


    # def test_get(self):
    #     val = LRU_Cache(10)

    #     val.set("val1", 1)
    #     self.assertEqual(val.get("val1"), 1)
    #     self.assertEqual(val.get("val1"), None)
    #     self.assertEqual(val.get("val1"), "bull1")
    #     self.assertRaises(KeyError, val.cache["val1"])
    #     self.assertRaises(KeyError, val.cache["val2"])
    #     self.assertRaises(KeyError, val.order["val1"])
    #     self.assertRaises(KeyError, val.order["val2"])

    #     val.set("val13", "bull2")
    #     val.set("val14", "bull2")
    #     val.set("val15", "bull2")
    #     val.set("val16", "bull2")
    #     self.assertRaises(KeyError, val.cache["val3"])
    #     self.assertRaises(KeyError, val.cache["val4"])
    #     self.assertRaises(KeyError, val.order["val3"])
    #     self.assertRaises(KeyError, val.order["val4"])
    #     self.assertRaises(KeyError, val.cache["val5"])
    #     self.assertRaises(KeyError, val.cache["val6"])
    #     self.assertRaises(KeyError, val.order["val5"])
    #     self.assertRaises(KeyError, val.order["val6"])

    #     self.assertEqual(val.get("val13"), "bull2")
    #     self.assertEqual(val.get("val14"), "bull2")
    #     self.assertEqual(val.get("val15"), "bull2")
    #     self.assertEqual(val.get("val16"), "bull2")
