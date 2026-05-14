#!/usr/bin/env python3
import unittest

from pydantic import BaseModel

from chat_workflow.interactive_entity import InteractiveEntity


class TestInteractiveEntityBase(unittest.TestCase):
    def test_is_base_model_subclass(self):
        self.assertTrue(issubclass(InteractiveEntity, BaseModel))

    def test_has_validation_rules_attribute(self):
        self.assertTrue(hasattr(InteractiveEntity, "_validation_rules"))
        entity = InteractiveEntity()
        self.assertEqual(entity._validation_rules, "")

    def test_can_instantiate_directly(self):
        entity = InteractiveEntity()
        self.assertIsInstance(entity, InteractiveEntity)
        self.assertEqual(entity._validation_rules, "")

    def test_can_create_subclass_with_custom_fields(self):
        class MyEntity(InteractiveEntity):
            name: str = ""
            value: int = 0

        self.assertTrue(issubclass(MyEntity, InteractiveEntity))
        self.assertTrue(issubclass(MyEntity, BaseModel))

    def test_subclass_instantiation_with_field_values(self):
        class MyEntity(InteractiveEntity):
            name: str = ""
            value: int = 0

        entity = MyEntity(name="test", value=42)
        self.assertEqual(entity.name, "test")
        self.assertEqual(entity.value, 42)
        self.assertIsInstance(entity, InteractiveEntity)

    def test_validation_rules_can_be_set_on_subclass(self):
        class MyEntity(InteractiveEntity):
            _validation_rules: str = "name must be non-empty"

        entity = MyEntity()
        self.assertEqual(entity._validation_rules, "name must be non-empty")

    def test_validation_rules_default_on_subclass(self):
        class MyEntity(InteractiveEntity):
            name: str = ""

        entity = MyEntity()
        self.assertEqual(entity._validation_rules, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
