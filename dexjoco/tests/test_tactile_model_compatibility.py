from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from dexjoco.sim.tactile import (
    BIMANUAL_FINGERTIP_BODY_NAMES,
    FingertipContactExtractor,
    SINGLE_FINGERTIP_BODY_NAMES,
)


_XML_DIR = (
    Path(__file__).resolve().parents[1]
    / "dexjoco"
    / "sim"
    / "envs"
    / "xmls"
)


class TactileModelCompatibilityTest(unittest.TestCase):
    def test_all_task_models_have_stable_collision_fingertips(self):
        xml_paths = sorted(_XML_DIR.glob("arena_arm_hand_*.xml"))
        self.assertEqual(len(xml_paths), 11)

        for xml_path in xml_paths:
            with self.subTest(model=xml_path.name):
                model = mujoco.MjModel.from_xml_path(str(xml_path))
                data = mujoco.MjData(model)
                names = (
                    BIMANUAL_FINGERTIP_BODY_NAMES
                    if mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        BIMANUAL_FINGERTIP_BODY_NAMES[0],
                    )
                    >= 0
                    else SINGLE_FINGERTIP_BODY_NAMES
                )
                extractor = FingertipContactExtractor(model, data, names)

                for fingertip_index, body_name in enumerate(names):
                    body_id = model.body(body_name).id
                    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
                    collision_geom_ids = geom_ids[
                        (model.geom_contype[geom_ids] != 0)
                        | (model.geom_conaffinity[geom_ids] != 0)
                    ]
                    self.assertGreater(collision_geom_ids.size, 0)
                    np.testing.assert_array_equal(
                        extractor._geom_to_fingertip[collision_geom_ids],
                        fingertip_index,
                    )


if __name__ == "__main__":
    unittest.main()
