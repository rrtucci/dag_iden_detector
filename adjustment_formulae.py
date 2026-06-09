from potentials.DiscreteCondPot import *
from graphs.BayesNet import *

"""
All the methods in this file return a conditional probability P(y|x) or P(
y|x, z) dictated by an adjustment formula (AF).

The AFs considered are

1. backdoor AF
2. frontdoor AF
3. Napkin1 AF
4. Napkin2 AF (fails to reproduce P(y|do(x))
5. Napkin3 AF (fails to reproduce P(y|do(x))
6. Napkin4 AF (fails to reproduce P(y|do(x))

1 to 5 return a P(y|x). 6 returns a P(y|x, z)

"""


def get_backdoor_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    nd_z = bnet.get_node_named('z')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    pot_xz = full_pot.get_new_marginal([nd_x, nd_z])
    pot_z = full_pot.get_new_marginal([nd_z])
    pot_xy = ((full_pot / pot_xz) * pot_z).get_new_marginal([nd_x, nd_y])
    arr_xy = pot_xy.pot_arr
    pot_y_bar_x = DiscreteCondPot(False,
                                  [nd_x, nd_y],
                                  arr_xy)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr
    return prob_y_bar_x


def get_frontdoor_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    # nd_h = bnet.get_node_named('h')
    nd_m = bnet.get_node_named('m')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    pot_mxy = full_pot.get_new_marginal([nd_m, nd_x, nd_y])
    pot_mx = full_pot.get_new_marginal([nd_m, nd_x])
    pot_x = pot_mx.get_new_marginal([nd_x])

    pot_my = (pot_mxy * pot_x / pot_mx).get_new_marginal([nd_m, nd_y])
    pot_xy = (pot_my * pot_mx / pot_x).get_new_marginal([nd_x, nd_y])
    arr_xy = pot_xy.pot_arr
    pot_y_bar_x = DiscreteCondPot(False,
                                  [nd_x, nd_y],
                                  arr_xy)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr
    return prob_y_bar_x


def get_napkin1_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    nd_w = bnet.get_node_named('w')
    nd_z = bnet.get_node_named('z')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    pot_wzxy = full_pot.get_new_marginal([nd_w, nd_z, nd_x, nd_y])
    pot_wz = pot_wzxy.get_new_marginal([nd_w, nd_z])
    pot_w = pot_wz.get_new_marginal([nd_w])
    pot_z = pot_wz.get_new_marginal([nd_z])

    final_pot = pot_wzxy * pot_w * pot_z / pot_wz

    arr_xy = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
    pot_y_bar_x = DiscreteCondPot(False,
                                  [nd_x, nd_y],
                                  arr_xy)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr
    return prob_y_bar_x


def get_napkin2_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    nd_z = bnet.get_node_named('z')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    pot_zxy = full_pot.get_new_marginal([nd_z, nd_x, nd_y])
    pot_zx = pot_zxy.get_new_marginal([nd_z, nd_x])
    pot_z = pot_zx.get_new_marginal([nd_z])

    final_pot = (pot_zxy / pot_zx) * pot_z

    arr_y_bar_x = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
    pot_y_bar_x = DiscreteCondPot(False,
                                  [nd_x, nd_y],
                                  arr_y_bar_x)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr
    return prob_y_bar_x


def get_napkin3_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    nd_w = bnet.get_node_named('w')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    pot_wxy = full_pot.get_new_marginal([nd_w, nd_x, nd_y])
    pot_wx = pot_wxy.get_new_marginal([nd_w, nd_x])
    pot_w = pot_wx.get_new_marginal([nd_w])

    final_pot = (pot_wxy / pot_wx) * pot_w

    arr_y_bar_x = final_pot.get_new_marginal([nd_x, nd_y]).pot_arr
    pot_y_bar_x = DiscreteCondPot(False,
                                  [nd_x, nd_y],
                                  arr_y_bar_x)
    pot_y_bar_x.normalize_self()
    prob_y_bar_x = pot_y_bar_x.pot_arr
    return prob_y_bar_x


def get_napkin4_adjustment_prob(bnet, full_pot):
    """

    Parameters
    ----------
    bnet: BayesNet
    full_pot: Potential

    Returns
    -------
    np.array

    """
    nd_w = bnet.get_node_named('w')
    nd_z = bnet.get_node_named('z')
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    pot_wzxy = full_pot.get_new_marginal([nd_w, nd_z, nd_x, nd_y])
    pot_wz = pot_wzxy.get_new_marginal([nd_w, nd_z])
    pot_w = pot_wz.get_new_marginal([nd_w])
    # print("nnmf", pot_w.pot_arr)

    final_pot = (pot_wzxy / pot_wz) * pot_w

    arr_xzy = final_pot.get_new_marginal([nd_x, nd_z, nd_y]).pot_arr
    pot_y_bar_x_z = DiscreteCondPot(False,
                                    [nd_x, nd_z, nd_y],
                                    arr_xzy)
    pot_y_bar_x_z.normalize_self()
    prob_y_bar_x_z = pot_y_bar_x_z.pot_arr
    return prob_y_bar_x_z
