import jax

def order1_eps_fn(meta):
    eps, _ = meta
    return eps

def order2_eps_fn(meta):
    eps, pred_eps = meta
    return 1.0 / 2 * (
        3 * eps \
        - 1 * pred_eps[0]
    )

def order3_eps_fn(meta):
    eps, pred_eps = meta
    return 1.0 / 12 * (
            23 * eps \
            - 16 * pred_eps[0] \
            + 5 * pred_eps[1] \
        )

def order4_eps_fn(meta):
    eps, pred_eps = meta
    return 1.0 / 24 * (
            55 * eps \
            - 59 * pred_eps[0] \
            + 37 * pred_eps[1] \
            - 9 * pred_eps[2]
        )

multistep_eps_fn = [order1_eps_fn,
                       order2_eps_fn,
                       order3_eps_fn,
                       order4_eps_fn,
                      ]

def get_ipndm_eps_fn(eps_fn, timesteps):
    def _fn(x, i, pred_eps):
        t = timesteps[i]
        eps = eps_fn(x, t)
        cur_eps = jax.lax.switch(
            i,
            multistep_eps_fn,
            (eps, pred_eps)
        )
        return cur_eps, eps
    return _fn


def get_pndm_order4_eps(eps_fn, timesteps):
    def _fn(x, i, pred_eps):
        t = timesteps[i]
        eps = eps_fn(x, t)
        cur_eps = order4_eps_fn((eps, pred_eps))
        return cur_eps, eps
    return _fn

def get_runge_kutta_eps(eps_fn, transfer_fn, timesteps):
    def _fn(x, i, pred_eps):
        del pred_eps
        t1,t3 = timesteps[i], timesteps[i+1]
        t2 = (t1 + t3) / 2.0

        e_1 = eps_fn(x, t1)
        x_2  = transfer_fn(x, t1, t2, e_1)

        e_2 = eps_fn(x_2, t2)
        x_3 = transfer_fn(x, t1, t2, e_2)

        e_3 = eps_fn(x_3, t2)
        x_4 = transfer_fn(x, t1, t3, e_3)

        e_4 = eps_fn(x_4, t3)

        cur_eps = 1.0 / 6 * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
        return cur_eps, e_1
    return _fn


def get_pndm_eps_fn(eps_fn, transfer_fn, timesteps):
    _runge_kutta_eps_fn = get_runge_kutta_eps(eps_fn, transfer_fn, timesteps)
    _order_4_eps_fn = get_pndm_order4_eps(eps_fn, timesteps)
    def runge_kutta_wrap_fn(meta):
        x, i, pred_eps = meta
        return _runge_kutta_eps_fn(x, i, pred_eps)
    def order_4_wrap_fn(meta):
        x, i, pred_eps = meta
        return _order_4_eps_fn(x, i, pred_eps)
    _pndm_eps_fn = [runge_kutta_wrap_fn,
               runge_kutta_wrap_fn,
               runge_kutta_wrap_fn,
               order_4_wrap_fn,
              ]
    def _fn(x, i, pred_eps):
        cur_eps = jax.lax.switch(
            i,
            _pndm_eps_fn,
            (x, i, pred_eps)
        )
        return cur_eps
    return _fn