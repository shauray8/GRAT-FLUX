## GRAT-X for FLUX

Hacked GRAT-X onto FLUX.1-dev. It’s a criss-cross attention mechanism that should be _fast_ but isn’t atleast in my case. 

![img7](https://github.com/user-attachments/assets/56a8841b-2cfd-4d0c-8d57-3585612d49b1)

* Groups 3072 image tokens into [8x8] patches (6x8 groups).
* Each query group attends to its row (m=p) or column (n=q).
* Complexity: O(6⋅3072+8⋅3072)≈43K, vs. full attention’s ( 9.44M ).

```py
git clone https://github.com/shauray8/GRAT-FLUX && cd GRAT-FLUX
# install all the requirements [pkgs needed for normal FLUX inference]
python inference.py
```
Should spit out hummingbird(s) and some beautiful flowers. Needs:

* Python 3.8+, PyTorch 2.0+, Diffusers (pip install diffusers)
* GPU (tested on A40)
<hr>

### Why No Big Speedup?
GRAT-X should crush full attention, but I got 1-2s faster on 30 steps. Tried group sizes 8, 16, 32—meh. Why?

* Default SDPA is a beast ! Flex with a custom mask probably does not even come close to the performance (needs to be confirmed) 
* Clustering’s reshapes and einsum ops add overhead, even if cache-optimized.
