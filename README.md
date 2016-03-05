# morphological-reinflection

Usage:

    pycnn_faruqui_inflection.py [--input=INPUT] [--hidden=HIDDEN] [--epochs=EPOCHS] [--layers=LAYERS] [--optimization=OPTIMIZATION] TRAIN_PATH TEST_PATH RESULTS_PATH SIGMORPHON_PATH...

Arguments:
*  TRAIN_PATH    train file path (sigmorphon format)
*  TEST_PATH     test file path (sigmorphon format)
*  RESULTS_PATH  results file to be written
*  SIGMORPHON_PATH   sigmorphon root containing data, src dirs

Options:
*  -h --help                     show this help message and exit
*  --input=INPUT                 input vector dimensions
*  --hidden=HIDDEN               hidden layer dimensions
*  --epochs=EPOCHS               amount of training epochs
*  --layers=LAYERS               amount of layers in lstm network
*  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM

For example:

    python pycnn_faruqui_inflection.py --input=120 --hidden=120 --epochs=100 --layers=2 --optimization=ADAM /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-train /Users/roeeaharoni/research_data/sigmorphon2016-master/data/navajo-task1-dev /Users/roeeaharoni/Dropbox/phd/research/morphology/inflection_generation/results/navajo_results.txt /Users/roeeaharoni/research_data/sigmorphon2016-master/
