
## Notes on reproducing

With the model weights downloaded from Google Drive and unpacked into the `vit_model` directory below:

```
python test.py -w ../vit_model/vit_finetuned_Bal_CE_lr5e-05_epochs10 -o out -m 2 -f ../../plankton_ml/tests/fixtures/test_images/
```

OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory ../vit_model/vit_finetuned_Bal_CE_lr5e-05_epochs10.

With the 3-class Resnet18 weights and the model_version 1 ("combined) we see output:
```
python test.py -w ../ResNet_18_3classes_RGB.pth -o out.csv -m 1 -f ../../plankton_ml/tests/fixtures/test_images/
```

With `-m 0` different `size mismatch` errors on model loading for both flavours



