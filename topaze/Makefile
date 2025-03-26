

.PHONY: clean status visu

status:
	ccc_mpp -u lecomtej

run:
	ccc_msub ./ccc_train.sh

clean:
	rm -rf *.o *.e mlruns

visu:
	ccc_visu virtual -A ifp00083 -t 7200 -p a100
