import jax
import jax.numpy as jnp

_eps_coef_fns = {}

def register_eps_fn(order=None):
  def _fn(func):
    _eps_coef_fns[order] = func
    return func
  return _fn

def get_eps_fn(order):
  return _eps_coef_fns[order]

def get_eps_coef_worker_fn(sde):
  def _worker(t_start, t_end, num_item):
    dt = (t_end - t_start) / num_item
    
    t_inter = jnp.linspace(t_start, t_end, num_item, endpoint=False)
    psi_coef = sde.psi(t_inter, t_end)
    integrand = sde.eps_integrand(t_inter)

    return psi_coef * integrand, t_inter, dt
  return _worker

def get_eps_coef_order1_seg_fn(sde):
  _eps_coef_worker = get_eps_coef_worker_fn(sde)
  def _eps_coef_order1_seg(t_1, t_0, order=1,num_item=10000):
    integrand, _, dt = _eps_coef_worker(t_1, t_0, num_item)
    order1_coef = jnp.sum(integrand) * dt
    return jax.lax.cond(order > 1, lambda _: 0.0, lambda _: order1_coef, _)
  return _eps_coef_order1_seg

@register_eps_fn(order=1)
def get_eps_coef_order1(sde, timesteps):
  rtn = jnp.zeros((timesteps.shape[0]-1, 3))
  eps_coef_order1_seg_fn = get_eps_coef_order1_seg_fn(sde)
  t_start, t_end = timesteps[:-1], timesteps[1:]
  coef_eps0 = jax.vmap(eps_coef_order1_seg_fn, in_axes=(0,0), out_axes=0)(t_start, t_end)
  return jnp.concatenate(
      [
          coef_eps0[:, None], rtn
      ],
      axis=1
  )

def order2_poly_coef_1_fn(meta):
    t_inter, t_2, t_1, t_0 = meta
    return (t_inter - t_2) / (t_1 - t_2)

def order2_poly_coef_2_fn(meta):
    t_inter, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_2 - t_1)

order2_poly_coef_fn = [lambda item: item[0],
                       order2_poly_coef_1_fn,
                       order2_poly_coef_2_fn,
                       lambda item: item[0] * 0.0
                      ]

def get_eps_coef_order2_seg_fn(sde):
  _eps_coef_worker = get_eps_coef_worker_fn(sde)
  def _eps_coef_order2_seg(t_2, t_1, t_0, coef_idx=1,num_item=10000):
    integrand, t_inter, dt = _eps_coef_worker(t_1, t_0, num_item)
    last_term = jax.lax.switch(coef_idx, order2_poly_coef_fn, (t_inter, t_2, t_1, t_0))
    return jnp.sum(integrand * last_term) * dt
  return _eps_coef_order2_seg

@register_eps_fn(order=2)
def get_eps_coef_order2(sde, timesteps):
  rtn = jnp.zeros((timesteps.shape[0]-1, 2))
  eps_coef_order1_sef_fn = get_eps_coef_order1_seg_fn(sde)

  first_eps0,first_eps1 = jax.vmap(eps_coef_order1_sef_fn, (None, None, 0), 0)(timesteps[0], timesteps[1], jnp.asarray([1,2]))

  eps_coef_order2_seg_fn = get_eps_coef_order2_seg_fn(sde)
  t_0,t_1,t_2 = timesteps[2:], timesteps[1:-1], timesteps[:-2]
  rest_eps0, rest_eps1 = jax.vmap(
      jax.vmap(eps_coef_order2_seg_fn, in_axes=(0,0,0,None), out_axes=0),
        in_axes = (None, None,None, 0), out_axes=0
    )(t_2, t_1, t_0, jnp.asarray([1,2]))
    
  coef_0 = jnp.append(first_eps0, rest_eps0)
  coef_1 = jnp.append(first_eps1, rest_eps1)
  return jnp.concatenate(
      [
          coef_0[:, None], coef_1[:, None], rtn
      ],
      axis=1
  )

def order3_poly_coef_1_fn(meta):
    t_inter, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_2) / (t_1 - t_2) * (t_inter - t_3) / (t_1 - t_3)

def order3_poly_coef_2_fn(meta):
    t_inter, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_2 - t_1) * (t_inter - t_3) / (t_2 - t_3)

def order3_poly_coef_3_fn(meta):
    t_inter, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_3 - t_1) * (t_inter - t_2) / (t_3 - t_2)

order3_poly_coef_fn = [lambda item: item[0],
                       order3_poly_coef_1_fn,
                       order3_poly_coef_2_fn,
                       order3_poly_coef_3_fn,
                       lambda item: item[0] * 0.0,
                      ]

def get_eps_coef_order3_seg_fn(sde):
  _eps_coef_worker = get_eps_coef_worker_fn(sde)
  def _eps_coef_order3_seg(t_3, t_2, t_1, t_0, coef_idx=1,num_item=10000):
    integrand, t_inter, dt = _eps_coef_worker(t_1, t_0, num_item)
    last_term = jax.lax.switch(coef_idx, order3_poly_coef_fn, (t_inter, t_3, t_2, t_1, t_0))
    return jnp.sum(integrand * last_term) * dt
  return _eps_coef_order3_seg

@register_eps_fn(order=3)
def get_eps_coef_order3(sde, timesteps):
  rtn = jnp.zeros((timesteps.shape[0]-1, 1))

  eps_coef_order1_sef_fn = get_eps_coef_order1_seg_fn(sde)
  first_eps0,first_eps1,first_eps2 = jax.vmap(eps_coef_order1_sef_fn, (None, None, 0), 0)(timesteps[0], timesteps[1], jnp.asarray([1,2,3]))

  eps_coef_order2_seg_fn = get_eps_coef_order2_seg_fn(sde)
  second_eps0, second_eps1,second_eps2 = jax.vmap(
      eps_coef_order2_seg_fn, (None, None, None, 0), 0
  )(timesteps[0], timesteps[1], timesteps[2], jnp.asarray([1,2, 3]))

  eps_coef_order3_seg_fn = get_eps_coef_order3_seg_fn(sde)
  t_0,t_1,t_2,t_3 = timesteps[3:], timesteps[2:-1], timesteps[1:-2], timesteps[0:-3]
  rest_eps0, rest_eps1, rest_eps2 = jax.vmap(
      jax.vmap(eps_coef_order3_seg_fn, in_axes=(0,0,0,0,None), out_axes=0),
        in_axes = (None, None,None, None, 0), out_axes=0
    )(t_3, t_2, t_1, t_0, jnp.asarray([1,2, 3]))
    

  coef_0 = jnp.append(jnp.asarray([first_eps0, second_eps0]), rest_eps0)
  coef_1 = jnp.append(jnp.asarray([first_eps1, second_eps1]), rest_eps1)
  coef_2 = jnp.append(jnp.asarray([first_eps2, second_eps2]), rest_eps2)
  return jnp.concatenate(
      [
          coef_0[:, None], coef_1[:, None], coef_2[:, None],rtn
      ],
      axis=1
  )

def order4_poly_coef_1_fn(meta):
    t_inter, t_4, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_2) / (t_1 - t_2) * (t_inter - t_3) / (t_1 - t_3) * (t_inter - t_4) / (t_1 - t_4)

def order4_poly_coef_2_fn(meta):
    t_inter, t_4, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_2 - t_1) * (t_inter - t_3) / (t_2 - t_3) * (t_inter - t_4) / (t_2 - t_4)

def order4_poly_coef_3_fn(meta):
    t_inter, t_4, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_3 - t_1) * (t_inter - t_2) / (t_3 - t_2) * (t_inter - t_4) / (t_3 - t_4)

def order4_poly_coef_4_fn(meta):
    t_inter, t_4, t_3, t_2, t_1, t_0 = meta
    return (t_inter - t_1) / (t_4 - t_1) * (t_inter - t_2) / (t_4 - t_2) * (t_inter - t_3) / (t_4 - t_3)

order4_poly_coef_fn = [lambda item: item[0],
                       order4_poly_coef_1_fn,
                       order4_poly_coef_2_fn,
                       order4_poly_coef_3_fn,
                       order4_poly_coef_4_fn,
                       lambda item: item[0] * 0.0,
                      ]

def get_eps_coef_order4_seg_fn(sde):
  _eps_coef_worker = get_eps_coef_worker_fn(sde)
  def _eps_coef_order4_seg(t_4, t_3, t_2, t_1, t_0, coef_idx=1,num_item=10000):
    integrand, t_inter, dt = _eps_coef_worker(t_1, t_0, num_item)
    last_term = jax.lax.switch(coef_idx, order4_poly_coef_fn, (t_inter, t_4, t_3, t_2, t_1, t_0))
    return jnp.sum(integrand * last_term) * dt
  return _eps_coef_order4_seg

@register_eps_fn(order=4)
def get_eps_coef_order4(sde, timesteps):
  eps_coef_order1_seg_fn = get_eps_coef_order1_seg_fn(sde)
  eps_coef_order2_seg_fn = get_eps_coef_order2_seg_fn(sde)
  eps_coef_order3_seg_fn = get_eps_coef_order3_seg_fn(sde)
  eps_coef_order4_seg_fn = get_eps_coef_order4_seg_fn(sde)

  first_coef = jax.vmap(eps_coef_order1_seg_fn, (None, None, 0), 0)(timesteps[0], timesteps[1], jnp.asarray([1,2, 3, 4])) # 4
  second_coef = jax.vmap(eps_coef_order2_seg_fn, (None, None, None, 0), 0)(timesteps[0], timesteps[1], timesteps[2], jnp.asarray([1,2, 3, 4])) # 4
  third_coef = jax.vmap(eps_coef_order3_seg_fn, (None, None, None, None, 0), 0)(timesteps[0], timesteps[1], timesteps[2], timesteps[3], jnp.asarray([1,2, 3, 4])) # 4

  t_0,t_1,t_2,t_3,t_4 = timesteps[4:], timesteps[3:-1], timesteps[2:-2], timesteps[1:-3], timesteps[0:-4]
  rest_coef = jax.vmap(
    jax.vmap(eps_coef_order4_seg_fn, in_axes=(0, 0,0,0,0,None), out_axes=0),
    in_axes = (None, None,None, None, None, 0), out_axes=0
  )(t_4, t_3, t_2, t_1, t_0, jnp.asarray([1,2,3,4])) # 4 * (t-4)

  return jnp.concatenate(
      [
          jnp.asarray(first_coef).reshape(-1,4),
          jnp.asarray(second_coef).reshape(-1,4),
          jnp.asarray(third_coef).reshape(-1,4),
          jnp.asarray(rest_coef).T,
      ],
      axis=0
  )


def ei_x_step(x, ei_coef, new_eps, eps_pred):
  x_coef, eps_coef = ei_coef[0], ei_coef[1:]
  full_eps = jnp.concatenate([new_eps[None], eps_pred])
  eps_term = jnp.einsum("i,i...->...", eps_coef, full_eps)
  return x_coef * x + eps_term, full_eps[:-1]