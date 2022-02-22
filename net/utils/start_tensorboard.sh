# Start tensorboard.

if [ "$1" = "" ]; then
  echo "Please specify `logdir`."
  exit
fi

echo -e "WARNING: Use command \e[1;33m 'ssh luyuchen@10.201.0.146 -p 22 -L 6006:localhost:6006'\e[0m on localhost!"

tensorboard --logdir="$1" --bind_all

