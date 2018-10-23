#########################################################################
# File Name: tensor.sh
# Author: hanlu
# mail: hanlu@shu.edu.cn
# Created Time: 2018年10月16日 星期二 10时33分55秒
#########################################################################
#!/bin/bash
cd /data/hanlu
for i in 1 2
do
	make
	./tensor
	cd V100\(16\)
	make
	./tensor
	cd ..
	cd V100\(32\)
	make
	./tensor
	cd ..
done

