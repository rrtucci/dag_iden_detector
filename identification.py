from adjustment_formulae import *
from pprint import pprint


def create_random_bnet(nodes,
                       arrows,
                       nd_to_size):
    """
    This method returns a BayesNet object whose structure is given by
    'nodes' and 'arrows'. The TPM (transition probability matrix) for
    each node is created at random, with the only constraint being that
    the number of states of each node be as specified by the input
    'nd_to_size'.

    Parameters
    ----------
    nodes: list[str]
        example: ['a', 'b', 'c']
    arrows: list[tuple[str, str]]
        example: [('a', 'b'), ('a', 'c')]
    nd_to_size: dict[str, int]
        dictionary mapping node name to its size (i.e., the number of
        values or states)

    Returns
    -------
    BayesNet

    """
    bnet_nodes = []
    found_x = False
    found_y = False
    for name in nodes:
        if name == "x":
            found_x = True
        if name == "y":
            found_y = True
    assert found_x and found_y, \
        "can't find 'x' or 'y' in list of node names"

    for k, node_name in enumerate(nodes):
        nd = BayesNode(k, name=node_name)
        nd.size = nd_to_size[node_name]
        bnet_nodes.append(nd)
    bnet = BayesNet(set(bnet_nodes))
    for arrow in arrows:
        pa_nd = bnet.get_node_named(arrow[0])
        child_nd = bnet.get_node_named(arrow[1])
        child_nd.add_parent(pa_nd)

    # print("ccvv", bnet.nodes)
    for nd in bnet_nodes:
        ord_nodes = list(nd.parents) + [nd]
        # print("llkjh", ord_nodes)
        nd.potential = DiscreteCondPot(False, ord_nodes)
        nd.potential.set_to_random()
        nd.potential.normalize_self()
    # print("aadf", bnet)
    return bnet


def randomize_these_nodes(bnet, some_node_names):
    for name in some_node_names:
        nd = bnet.get_node_named(name)
        nd.potential.set_to_random()
        nd.potential.normalize_self()


def get_default_nd_to_size(dot_file, hidden_nd_names):
    nodes, _ = DotTool.read_dot_file(dot_file)
    nd_to_size = {}
    for nd in nodes:
        nd_to_size[nd] = 2
    for nd_name in hidden_nd_names:
        assert nd_name in nodes, (f"hidden node '{nd_name}'"
                                  f" not in full node list")
        nd_to_size[nd_name] = 3
    return nd_to_size


def calc_ampu_and_full_pots(bnet):
    node_list = list(bnet.nodes)

    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')

    ampu_pot = None
    for nd in node_list:
        if nd != nd_x:
            if not ampu_pot:
                ampu_pot = nd.potential
            else:
                ampu_pot = ampu_pot * nd.potential
    full_pot = ampu_pot * nd_x.potential
    return ampu_pot, full_pot


def calc_ampu_and_full_prob_y_bar_x(bnet,
                                    ampu_pot,
                                    full_pot):
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    ampu_prob_y_bar_x = None
    full_prob_y_bar_x = None
    for final_pot in [ampu_pot, full_pot]:
        arr_xy = final_pot.get_new_marginal(
            [nd_x, nd_y]).pot_arr
        pot_y_bar_x = DiscreteCondPot(
            False,
            [nd_x, nd_y],
            arr_xy)

        pot_y_bar_x.normalize_self()
        if final_pot == ampu_pot:
            ampu_prob_y_bar_x = pot_y_bar_x.pot_arr
        elif final_pot == full_pot:
            full_prob_y_bar_x = pot_y_bar_x.pot_arr
        else:
            assert None

    return ampu_prob_y_bar_x, full_prob_y_bar_x


def print_all_prob_y_bar_x(dot_file,
                           hidden_nd_names,
                           nd_to_size=None,
                           verbose=True,
                           adj_version=1,
                           num_bnet_samples=2):
    if not nd_to_size:
        nd_to_size = get_default_nd_to_size(dot_file, hidden_nd_names)
    nodes, arrows = DotTool.read_dot_file(dot_file)

    bnet = create_random_bnet(
        nodes,
        arrows,
        nd_to_size)

    if num_bnet_samples == 1:
        bnet_strings = ["bnet1"]
    else:
        bnet_strings = ["bnet1", "bnet2"]
    for bnet_str in bnet_strings:
        if bnet_str == "bnet2":
            print("------------------------------")
            randomize_these_nodes(bnet, hidden_nd_names)
        print(f"Random {bnet_str}:")
        if verbose:
            print(bnet)
        ampu_pot, full_pot = calc_ampu_and_full_pots(bnet)
        ampu_prob_y_bar_x, full_prob_y_bar_x = \
            calc_ampu_and_full_prob_y_bar_x(bnet, ampu_pot, full_pot)
        print()
        print(f"full P(y|x) for {bnet_str}:")
        pprint(full_prob_y_bar_x)
        print()
        print(f"amputated P(y|x) for {bnet_str}: (REQUIRES RCT)")
        pprint(ampu_prob_y_bar_x)
        dotf_strings = ["back-door", "front-door", "napkin"]
        adj_id_to_adj_method = {
            "back-door1": get_backdoor_adjustment_prob,
            "front-door1": get_frontdoor_adjustment_prob,
            "napkin1": get_napkin1_adjustment_prob,
            "napkin2": get_napkin2_adjustment_prob,
            "napkin3": get_napkin3_adjustment_prob,
            "napkin4": get_napkin4_adjustment_prob,
            "napkin5": get_napkin5_adjustment_prob}
        adj_method = None
        for dotf_str in dotf_strings:
            if dotf_str in dot_file:
                adj_id = dotf_str + str(adj_version)
                adj_method = adj_id_to_adj_method[adj_id]
                break
        if adj_method:
            print()
            print(f"adjusted P(y|x) from full_pot for {bnet_str}:")
            pprint(adj_method(bnet, full_pot))
            print()
            print(f"adjusted P(y|x) from ampu_pot for {bnet_str}:"
                  f"(REQUIRES RCT)")
            pprint(adj_method(bnet, ampu_pot))


def calc_ampu_and_full_prob_y_bar_x_z(bnet,
                                      ampu_pot,
                                      full_pot,
                                      other_cond):
    assert isinstance(other_cond, str)
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    nd_z = bnet.get_node_named(other_cond)
    ampu_prob_y_bar_x_z = None
    full_prob_y_bar_x_z = None
    for final_pot in [ampu_pot, full_pot]:
        arr_zxy = final_pot.get_new_marginal(
            [nd_z, nd_x, nd_y]).pot_arr
        pot_y_bar_x_z = DiscreteCondPot(
            False,
            [nd_z, nd_x, nd_y],
            arr_zxy)

        pot_y_bar_x_z.normalize_self()
        if final_pot == ampu_pot:
            ampu_prob_y_bar_x_z = pot_y_bar_x_z.pot_arr
        elif final_pot == full_pot:
            full_prob_y_bar_x_z = pot_y_bar_x_z.pot_arr
        else:
            assert None

    return ampu_prob_y_bar_x_z, full_prob_y_bar_x_z


def print_all_prob_y_bar_x_z(dot_file,
                             hidden_nd_names,
                             other_cond,
                             nd_to_size=None,
                             verbose=True,
                             adj_version=1,
                             num_bnet_samples=2):
    assert isinstance(other_cond, str)
    if not nd_to_size:
        nd_to_size = get_default_nd_to_size(dot_file, hidden_nd_names)
    nodes, arrows = DotTool.read_dot_file(dot_file)

    bnet = create_random_bnet(
        nodes,
        arrows,
        nd_to_size)

    if num_bnet_samples == 1:
        bnet_strings = ["bnet1"]
    else:
        bnet_strings = ["bnet1", "bnet2"]
    for bnet_str in bnet_strings:
        if bnet_str == "bnet2":
            print("------------------------------")
            randomize_these_nodes(bnet, hidden_nd_names)
        print(f"Random {bnet_str}:")
        if verbose:
            print(bnet)
        ampu_pot, full_pot = calc_ampu_and_full_pots(bnet)
        ampu_prob_y_bar_x_z, full_prob_y_bar_x_z = \
            calc_ampu_and_full_prob_y_bar_x_z(
                bnet, ampu_pot, full_pot, other_cond)
        print()
        print(f"full P(y|x, z) for {bnet_str}:")
        pprint(full_prob_y_bar_x_z)
        print()
        print(f"amputated P(y|x, z) for {bnet_str}: (REQUIRES RCT)")
        pprint(ampu_prob_y_bar_x_z)
        dotf_strings = ["napkin"]
        adj_id_to_adj_method = {
            "napkin6": get_napkin6_adjustment_prob}
        adj_method = None
        for dotf_str in dotf_strings:
            if dotf_str in dot_file:
                adj_id = dotf_str + str(adj_version)
                adj_method = adj_id_to_adj_method[adj_id]
                break
        if adj_method:
            print()
            print(f"adjusted P(y|x, "
                  f"{other_cond}) from full_pot for {bnet_str}:")
            pprint(adj_method(bnet, full_pot))
            print()
            print(f"adjusted P(y|x, "
                  f"{other_cond}) from ampu_pot for {bnet_str}:"
                  f"(REQUIRES RCT)")
            pprint(adj_method(bnet, ampu_pot))


if __name__ == "__main__":
    def main_backdoor(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/back-door.dot",
                               hidden_nd_names=[],
                               verbose=verbose)


    def main_frontdoor(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/front-door.dot",
                               hidden_nd_names=["h"],
                               verbose=verbose)


    def main_napkin1(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=1,
                               verbose=verbose)


    def main_napkin2(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=2,
                               verbose=verbose)


    def main_napkin3(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=3,
                               verbose=verbose)


    def main_napkin4(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=4,
                               verbose=verbose)


    def main_napkin5(draw, verbose):
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=5,
                               verbose=verbose)

    def main_napkin6(draw, verbose):
        print_all_prob_y_bar_x_z(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               other_cond = "z",
                               adj_version=6,
                               verbose=verbose)



    # main_backdoor(False, False)
    # main_frontdoor(False, False)
    # main_napkin1(False, False)
    # main_napkin2(False, False)
    # main_napkin3(False, False)
    # main_napkin4(False, False)
    # main_napkin5(False, False)
    main_napkin6(False, False)
