import albumentations as A

transforms = A.compose([
	A.HorizontaFlip(p=0.2),
	A.RandomBrightnessContrast(p=0.2),
	
])