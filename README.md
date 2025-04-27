# factorization_v2

Will factor 150 bits in a couple minutes, usage:

python3 QS.py -keysize 150

It is still wildly unoptimized. But I hope to address that in version 3.


Note: If you want to optimize this version yourself, rewriting it in C and implementing block lanczos instead of Guassian elimination would be a good start. Further improvements can then also be made in coefficients selection / adjustment such that small enough values are generated. For v3 I am hoping to construct smooths from the ground up.. but it's no easy task, we'll see.

Update: Next iteration is almost finished but wont be publically available. Pay me if you want access. Done sharing things for free. 

Update: Ergh.... I just realized, there is absolutely no point in finding squares. Since we are taking the GCD on quadratic coefficients in essence, and I already know how to calculate quadratic coefficients... which eliminates the need to find squares........... I don't know... I took a break for an entire week.. looked at it again this weekend... and suddenly saw it. lol. It's kind of obvious in retrospect. And since it is so obvious.. I will release it as v3 soon.... no point in trying to sell something that is already obvious. But after v3... I will only sell my work going forward. 
