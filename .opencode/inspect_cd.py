import inspect

from tensordict.nn import CompositeDistribution as CD

print("sig:", inspect.signature(CD.__init__))
print("=" * 60)
src = inspect.getsource(CD)
print(src[:6000])
