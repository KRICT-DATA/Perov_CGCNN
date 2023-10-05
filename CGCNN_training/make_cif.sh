SCRIPT_DIR=$(pwd)
mkdir $SCRIPT_DIR/cif_dir
mv $SCRIPT_DIR/atom_init.json $SCRIPT_DIR/cif_dir
DIR=`ls -d ../DFT_training_data/*/`
n=0
for d in ${DIR}; do
    elems=`echo $d | cut -d "/" -f 3`
	A=`echo $elems | cut -d '.' -f 1`
	B1=`echo $elems | cut -d '.' -f 2`
	B2=`echo $elems | cut -d '.' -f 3`
	B3=`echo $elems | cut -d '.' -f 4`
	B4=`echo $elems | cut -d '.' -f 5`
	X=`echo $elems | cut -d '.' -f 6`
	cd $SCRIPT_DIR/cif_dir
	cp ../POSCAR .
	sed -i "s/Cs/$A/g" POSCAR
	sed -i "s/Hf/$B1/g" POSCAR
	sed -i "s/W/$B2/g" POSCAR
	sed -i "s/Au/$B3/g" POSCAR
	sed -i "s/Pt/$B4/g" POSCAR
	sed -i "s/I/$X/g" POSCAR
	mv POSCAR CONTCAR
	python ../convert.py
	mv CONTCAR.cif $n.cif
	n=$(($n+1))
	rm CONTCAR
done
