# download data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ./data/shakespeare.txt

# train
python train.py ./data/shakespeare.txt

# generate
python generate.py shakespeare.pt --prime_str "Where"
