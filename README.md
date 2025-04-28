# factorization_v2

Will factor 150 bits in a couple minutes, usage:

python3 QS.py -keysize 150

It is still wildly unoptimized. But I hope to address that in version 3.


Note: If you want to optimize this version yourself, rewriting it in C and implementing block lanczos instead of Guassian elimination would be a good start. Further improvements can then also be made in coefficients selection / adjustment such that small enough values are generated. For v3 I am hoping to construct smooths from the ground up.. but it's no easy task, we'll see.

Update: Next iteration is almost finished but wont be publically available. Pay me if you want access. Done sharing things for free. 

