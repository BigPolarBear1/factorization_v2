# factorization_v2

Will factor 150 bits in a couple minutes, usage:

python3 QS.py -keysize 150

It is still wildly unoptimized. But I hope to address that in version 3.


Note: If you want to optimize this version yourself, rewriting it in C and implementing block lanczos instead of Guassian elimination would be a good start. Further improvements can then also be made in coefficients selection / adjustment such that small enough values are generated. For v3 I am hoping to construct smooths from the ground up.. but it's no easy task, we'll see.

Update: A few months since releasing this... I am now 100% sure square finding can be completely bypassed and you should be able to work with just coefficients instead.

If N = 4387 (p=41 and q=107)

x<sup>2</sup> + y<sub>0</sub>x + N = x<sup>2</sup> + y<sub>1</sub>x - N

Which solves as (see v1 paper) (-41)<sup>2</sup> + 148 * (-41) + 4387 = (-41)<sup>2</sup> -66 * (-41) - 4387

Both sides share the same root but different coefficient.
We can construct coefficient candidates mod p<sub>i</sub> as described in the paper. Each coefficient mod p will have 2 possible roots.
Say we have a coefficient 148 and a root -41, we simply take the derivative of the above equation: 2 * (-41) + 148 = 66, which yield the coefficient for the other side. And then we can just proceed taking the GCD(148-66,4387).
You should be able to get this approach working with any coefficient and root in mod m, where m is constructed from p<sub>i</sub>

I hope to release more details very soon in the v3 repo...I know it can be done. I will succeed. And I will keep going until factorization is completely broken. I'm never stopping. I'm going to fix this shit world, and I'm going to get justice for what MSFT did to my former manager. Shouldn't have fired the best manager in the world.
