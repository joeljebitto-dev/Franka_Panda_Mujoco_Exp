# Assets

Place MuJoCo Menagerie assets here so the default Panda model path exists:

```text
assets/mujoco_menagerie/franka_emika_panda/scene.xml
```

Copy or clone MuJoCo Menagerie into this directory before running the viewer.

Sparse checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_emika_panda
```
