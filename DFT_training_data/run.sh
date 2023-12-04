SCRIPT_DIR=$(pwd)
A_elems=("Cs" "K" "Rb")
X_elems=("Br" "Cl" "I")
B_unarys=("Cd" "Ge" "Hg" "Pb" "Sn" "Zn")
B_binarys=("Cd-Ge" "Cd-Hg" "Cd-Pb" "Cd-Sn" "Cd-Zn" "Ge-Hg" "Ge-Pb" "Ge-Sn" "Ge-Zn" "Hg-Pb" "Hg-Sn" "Hg-Zn" "Pb-Sn" "Pb-Zn" "Sn-Zn")
B_ternarys=("Cd-Ge-Hg" "Cd-Ge-Pb" "Cd-Ge-Sn" "Cd-Ge-Zn" "Cd-Hg-Pb" "Cd-Hg-Sn" "Cd-Hg-Zn" "Cd-Pb-Sn" "Cd-Pb-Zn" "Cd-Sn-Zn" "Ge-Hg-Pb" "Ge-Hg-Sn" "Ge-Hg-Zn" "Ge-Pb-Sn" "Ge-Pb-Zn" "Ge-Sn-Zn" "Hg-Pb-Sn" "Hg-Pb-Zn" "Hg-Sn-Zn" "Pb-Sn-Zn")
B_quaternary=("Cd-Ge-Hg-Pb" "Cd-Ge-Hg-Sn" "Cd-Ge-Hg-Zn" "Cd-Ge-Pb-Sn" "Cd-Ge-Pb-Zn" "Cd-Ge-Sn-Zn" "Cd-Hg-Pb-Sn" "Cd-Hg-Pb-Zn" "Cd-Hg-Sn-Zn" "Cd-Pb-Sn-Zn" "Ge-Hg-Pb-Sn" "Ge-Hg-Pb-Zn" "Ge-Hg-Sn-Zn" "Ge-Pb-Sn-Zn" "Hg-Pb-Sn-Zn")
function run_cellrelax {
	mkdir $SCRIPT_DIR/$1.$2.$3.$4.$5.$6
	cd $SCRIPT_DIR/$1.$2.$3.$4.$5.$6
	ln -s $SCRIPT_DIR/INCAR .
	ln -s $SCRIPT_DIR/KPOINTS .
	cp $SCRIPT_DIR/job.sh .
	sed -i "2s/.*/#SBATCH --job-name=$1.$2.$3.$4.$5.$6/g" job.sh
	cp $SCRIPT_DIR/POSCAR .
	sed -i "s/Cs/$1/g" POSCAR
	sed -i "s/Hf/$2/g" POSCAR
	sed -i "s/W/$3/g" POSCAR
	sed -i "s/Au/$4/g" POSCAR
	sed -i "s/Pt/$5/g" POSCAR
	sed -i "s/I/$6/g" POSCAR
	rm POTCAR
	echo 103|vaspkit
	#qsub job.sh
	#sleep 0.1
}
function rerun_cellrelax {
	cd $SCRIPT_DIR/$1.$2.$3.$4.$5.$6
	cp CONTCAR POSCAR
	#qsub job.sh
	#sleep 0.1
}
function run_relax {
	cd $SCRIPT_DIR/$1.$2.$3.$4.$5.$6
	cp CONTCAR POSCAR
	#qsub job.sh
	#sleep 0.1
}
for A in "${A_elems[@]}"; do
	for X in "${X_elems[@]}"; do
		for unary in ${B_unarys[@]}; do
			B=$unary
			run_cellrelax $A $B $B $B $B $X 
		done
		for binary in ${B_binarys[@]}; do
			B1=`echo $binary | cut -d "-" -f1`
			B2=`echo $binary | cut -d "-" -f2`
			run_cellrelax $A $B1 $B2 $B2 $B2 $X #0.25
			run_cellrelax $A $B1 $B1 $B2 $B2 $X #0.5
			run_cellrelax $A $B1 $B2 $B1 $B2 $X #0.5
			run_cellrelax $A $B1 $B2 $B2 $B1 $X #0.5
			run_cellrelax $A $B1 $B1 $B1 $B2 $X #0.75
		done
		for ternary in ${B_ternarys[@]}; do
			B1=`echo $ternary | cut -d "-" -f1`
			B2=`echo $ternary | cut -d "-" -f2`
			B3=`echo $ternary | cut -d "-" -f3`
			run_cellrelax $A $B1 $B1 $B2 $B3 $X
			run_cellrelax $A $B1 $B2 $B1 $B3 $X
			run_cellrelax $A $B1 $B2 $B3 $B1 $X
			run_cellrelax $A $B2 $B2 $B1 $B3 $X
			run_cellrelax $A $B2 $B1 $B2 $B3 $X
			run_cellrelax $A $B2 $B1 $B3 $B2 $X
			run_cellrelax $A $B3 $B3 $B1 $B2 $X
			run_cellrelax $A $B3 $B1 $B3 $B2 $X
			run_cellrelax $A $B3 $B1 $B2 $B3 $X
		done
		for quaternary in ${B_quaternary[@]}; do
			B1=`echo $quaternary | cut -d "-" -f1`
			B2=`echo $quaternary | cut -d "-" -f2`
			B3=`echo $quaternary | cut -d "-" -f3`
			B4=`echo $quaternary | cut -d "-" -f4`
			run_cellrelax $A $B1 $B2 $B3 $B4 $X
			run_cellrelax $A $B1 $B4 $B3 $B2 $X
			run_cellrelax $A $B1 $B4 $B2 $B3 $X
			run_cellrelax $A $B1 $B3 $B4 $B2 $X
			run_cellrelax $A $B1 $B2 $B4 $B3 $X
			run_cellrelax $A $B1 $B3 $B2 $B4 $X
		done
	done
done
