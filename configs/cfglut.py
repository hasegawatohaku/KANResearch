KANELE_QAT_NAME = "ALL U-KAN + KANELE (QAT)"
KANELE_LUT_NAME = "ALL U-KAN + KANELE (LUT)"

KANELE_BITS = 8                   # activation precision = LUT address width
KANELE_QMIN = -16.0               # quantization domain of hidden activations;
KANELE_QMAX = 16.0                #
KANELE_INPUT_RANGE = (0.0, 1.0)   # input image range (pixel / 255)
KANELE_OUTPUT_RANGE = (KANELE_QMIN, KANELE_QMAX)  # quantization range of the logits
KANELE_TABLE_FRAC_BITS = 4        # fractional bits of the integer table entries;
                                  # 0 = format of the KANELÉ code (entry width = output width); each extra bit adds one bit to every table entry
KANELE_TARGET_SPARSITY = 0.20     # final fraction of pruned connections per layer
KANELE_PRUNING_START_EPOCH = 2
KANELE_CALIBRATION_IMAGES = 8     # training images used to score connections
KANELE_CHUNK_ELEMENTS = 1 << 22   # table values per chunk in LUT inference
KANELE_EXPORT_LUT = False          # save the integer lookup tables to a .json file