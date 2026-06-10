from adjustment_formulae import *
from pprint import pprint


def create_random_bnet(nodes,
                       arrows,
                       nd_to_size):
    """
    This method returns a BayesNet object whose structure is given by
    'nodes' and 'arrows'. The TPM (transition probability matrix, a.k.a.
    CPT, conditional probability table) for each node is created at random,
    with the only other constraint being that the number of states of each
    node be as specified by the input 'nd_to_size'.

    Parameters
    ----------
    nodes: list[str]
        example: ['a', 'b', 'c']
    arrows: list[tuple[str, str]]
        example: [('a', 'b'), ('a', 'c')], where ('a', 'b') means a->b
    nd_to_size: dict[str, int]
        dictionary mapping node name to its size (i.e., the number of values
        or states). In this method, `nd_to_size` must contain all the nodes.
        In all other methods in this file (for example,
        in `fill_node_to_size( )`), `nd_to_size` contains only special nodes
        which you don't want to have a default size. Default sizes are 2 for
        non-hidden nodes and 3 for hidden ones.

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
    """
    This method randomizes (i.e., replaces by random ones) the CPT (
    Conditional Probability Tables) of the nodes in the list
    `some_node_names` in the BayesNet object bnet.

    Parameters
    ----------
    bnet: BayesNet
        BayesNet object created by `create_random_bnet`. BayesNet or bnet
        stand for Bayesian network.
    some_node_names: list[str]
        This is a partial list of node names. Normally one uses for this a
        list of nodes that are hidden.

    Returns
    -------
    None

    """
    for name in some_node_names:
        nd = bnet.get_node_named(name)
        nd.potential.set_to_random()
        nd.potential.normalize_self()


def fill_nd_to_size(dot_file, hidden_nd_names, nd_to_size=None):
    """
    This method compiles the list of node names from the dot file
    `dot_file`. It then produces a default dict `nd_to_size1` that maps
    hidden nodes to 3 (i.e., they will have 3 states) and non-hidden ones to
    2. Then the method overrides `nd_to_size1` with the request of
    `nd_to_size` whenever they disagree. Finally, the method returns
    `nd_to_size1`

    Parameters
    ----------
    dot_file: str
        dot file (i.e., graphviz format) of the OP (Original Promise) bnet.
    hidden_nd_names: list[str]
        names of hidden nodes
    nd_to_size: dict[str, int] | None
        dict mapping each node name to its size. This input need only be a
        partial list of those nodes that you don't want to have default values.

    Returns
    -------
    list[str]

    """
    nodes, _ = DotTool.read_dot_file(dot_file)
    nd_to_size1 = {}
    for nd in nodes:
        nd_to_size1[nd] = 2
    for nd_name in hidden_nd_names:
        assert nd_name in nodes, (f"hidden node '{nd_name}'"
                                  f" not in full node list")
        nd_to_size1[nd_name] = 3
    if nd_to_size:
        for nd_name in nd_to_size:
            if nd_name in nd_to_size1:
                nd_to_size1[nd_name] = nd_to_size[nd_name]
    return nd_to_size1


def calc_ampu_and_full_pots(bnet):
    """
    This method calculates the probability distribution for

    (1) the "full" OP and

    (2) the "ampu" OP. By (ampu=amputated) OP, we mean a bnet whose
    arrows entering node "x" are amputated.

    These two probability distributions are outputted as 2 Potential
    objects, `full_pot` and `dot_pot`. Remember, a Potential can be thought
    of an arbitrary function f(x,y,z) where `x,y,z` are a partial list of
    the nodes of the OP bnet. A Potential is usually used to carry either a
    joint probability distribution like P(x,y,z) or a conditional one like
    P(z| x, y).

    Parameters
    ----------
    bnet: BayesNet

    Returns
    -------
    Potential, Potential

    """
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
    """
    This method takes the two Potentials `ampu_pot` and `full_pot` and
    calculates from these the two conditional probabilities
    `ampu_prob_y_bar_x`, `full_prob_y_bar_x'. The probabilities are of the
    type P(y|x) and expressed as numpy arrays. `ampu_prob_y_bar_x` equals P(
    y|do(x)).

    Parameters
    ----------
    bnet: BayesNet
    ampu_pot: Potential
    full_pot: Potential

    Returns
    -------
    np.array, np.array

    """
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
    """
    This method and the analogous one `print_all_prob_y_bar_x_z` are the
    only ones used in the jupyter notebooks. All others are meant to be
    internal. This method prints 4 things for each bnet.
    `num_bnet_samples=2` means the default is two bnets, but it will do only
    one bnet if you input `num_bnet_samples=1`

    1. full P(y|x) for OP

    2. amputated P(y|x) for OP. Calculating this in real life requires data
    from an RCT.

    3. adjusted P(y|x) from full_pot

    4. adjusted P(y|x) from ampu_pot. Calculating this in real life requires
    data from an RCT.


    Parameters
    ----------
    dot_file: str
        the dot file (i.e., graphviz format) of the OP bnet
    hidden_nd_names: list[str]
        the names of the hidden nodes
    nd_to_size: dict[str, int]| None
        a node name to node size dict for those nodes that you don't want to
        have default sizes.
    verbose: bool
        if this is set to True, the method prints also the CPT of each node
        of the bnet.
    adj_version: int
        the adjustment formula version. For the Napkin OP, there are
        currently 4 adjustment formulae that are tested.
    num_bnet_samples: int
        number of random bnets considered. This can be either 1 or 2.

    Returns
    -------
    None

    """
    nd_to_size = fill_nd_to_size(dot_file, hidden_nd_names, nd_to_size)

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
            "napkin3": get_napkin3_adjustment_prob}
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
                                      other_cond='z'):
    """
    The analogous method `calc_ampu_and_full_prob_y_bar_x` calculates P(
    y|x). This method calculates P(y| x, z) instead. If you don't want the
    name of the extra node to be "conditioned on" to default to 'z', you can
    tell the method the name of your alternative to "z" using the input
    variable `other_cond`


    Parameters
    ----------
    bnet: BayesNet
    ampu_pot: Potential
    full_pot: Potential
    other_cond: str

    Returns
    -------
    np.array, np.array

    """
    assert isinstance(other_cond, str)
    nd_x = bnet.get_node_named('x')
    nd_y = bnet.get_node_named('y')
    nd_z = bnet.get_node_named(other_cond)
    ampu_prob_y_bar_x_z = None
    full_prob_y_bar_x_z = None
    for final_pot in [ampu_pot, full_pot]:
        arr_xzy = final_pot.get_new_marginal(
            [nd_x, nd_z, nd_y]).pot_arr
        pot_y_bar_x_z = DiscreteCondPot(
            False,
            [nd_x, nd_z, nd_y],
            arr_xzy)

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
    """
    The analogous method `print_all_prob_y_bar_x` prints 4 probabilities P(
    y|x). This method prints 4 probabilities P(y| x, z) instead.

    Parameters
    ----------
    dot_file: str
    hidden_nd_names: list[str]
    other_cond: str
    nd_to_size: list[str, int]
    verbose: bool
    adj_version: int
    num_bnet_samples: int

    Returns
    -------
    None

    """
    assert isinstance(other_cond, str)
    nd_to_size = fill_nd_to_size(dot_file, hidden_nd_names, nd_to_size)
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
            "napkin4": get_napkin4_adjustment_prob}
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
        """
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x(dot_file="dot_atlas/back-door.dot",
                               hidden_nd_names=[],
                               verbose=verbose)


    def main_frontdoor(draw, verbose):
        """
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x(dot_file="dot_atlas/front-door.dot",
                               hidden_nd_names=["h"],
                               verbose=verbose)


    def main_napkin1(draw, verbose):
        """
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=1,
                               verbose=verbose)


    def main_napkin2(draw, verbose):
        """     
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=2,
                               verbose=verbose)


    def main_napkin3(draw, verbose):
        """
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x(dot_file="dot_atlas/napkin.dot",
                               hidden_nd_names=["u_1", "u_2"],
                               adj_version=3,
                               verbose=verbose)


    def main_napkin4(draw, verbose):
        """
        Parameters
        ----------
        draw: bool
        verbose: bool

        Returns
        -------
        None

        """
        print_all_prob_y_bar_x_z(dot_file="dot_atlas/napkin.dot",
                                 hidden_nd_names=["u_1", "u_2"],
                                 other_cond="z",
                                 nd_to_size={"z": 3},
                                 adj_version=4,
                                 verbose=verbose)


    # main_backdoor(False, False)
    # main_frontdoor(False, False)
    main_napkin1(False, False)
    # main_napkin2(False, False)
    # main_napkin3(False, False)
    # main_napkin6(False, False)
