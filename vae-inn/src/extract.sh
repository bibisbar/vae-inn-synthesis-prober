i=0
for img in cifar10/cifar10/train/**/*.png;
do
    cp $img cifar10/cifar10/train/$i.png
    ((i++))
done
