rm -r pk_cpp
rm -r __pycache__
python main.py -nw -s 200 -g 1200 -e 3.5 -N 1000000 -n 1 -p -ng 1
python main.py -nw -s 200 -g 1200 -e 3.5 -N 2000000 -n 1 -p -ng 2
python main.py -nw -s 200 -g 1200 -e 3.5 -N 3000000 -n 1 -p -ng 3
