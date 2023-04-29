#python main2.py -x 0.002 -C 0.04 -t 0.14 -m LaxWendroff
#python main.py -x 0.01 -s -1 -e 1 -C 0.8 -t 0.14 -m Upwind,LaxWendroff
python main.py -n 133 -s -1 -e 1 -C 0.8 -t 0.14 -m Upwind,LaxWendroff
