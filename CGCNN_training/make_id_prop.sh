## Usage: "bash make_id_prop.sh [property]"
## Replace [property] to one of the properties: decomp, bandgap, bandtype

target_dir="cif_dir"
if [ -e './${target_dir}/id_prop.csv' ]; then
rm ./${target_dir}/id_prop.csv
fi
touch ./${target_dir}/id_prop.csv
n=0
while read line; do
    target=`echo $line | cut -d ',' -f 2`
	echo $target
    echo "$n,$target" >> ./${target_dir}/id_prop.csv
    n=$(($n+1))
done < id_prop_for_$1.csv
