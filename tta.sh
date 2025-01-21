test_sets=('CIFAR-10-C' 'CIFAR-100-C' 'imagenet-c')
corruption_types=('gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')

for test_set in "${test_sets[@]}"; do
    for corruption_type in "${corruption_types[@]}"; do
        python BITTA.py --test_sets $test_set --corruption_type $corruption_type --gpu 0
    done
done