

for drop_rate in 0.2 0.4 0.8
do
# Clustering (table 2)
python main.py --name baron_mouse --drop_rate $drop_rate
python main.py --name mouse_es --drop_rate $drop_rate
python main.py --name mouse_bladder --drop_rate $drop_rate
python main.py --name zeisel --drop_rate $drop_rate
python main.py --name baron_human --drop_rate $drop_rate
done


# Clustering (table 2)
python main.py --name baron_mouse
python main.py --name mouse_es
python main.py --name mouse_bladder
python main.py --name zeisel
python main.py --name baron_human