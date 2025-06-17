class ElideFill(PatternRewrite):
    def __init__(self, p):
        super().__init__(p)
        self._return_if_success = True

    @work_on(Node, lambda node: node.op == "Fill")
    def match_and_rewrite(self, node: Node):
        if node.empty_use():
            return Failure()

        input = node.input_node(0)

        # NOTE: if input.op is Shape, then input is an 1-D tensor.
        if input.op not in ["Const", "Pack"]:
            return Failure()

        if input.op == "Pack":
            for inp in input.input_nodes():
                if inp.op != "Const":
                    return Failure()

        value = node.input_node(1)
        if value.op != "Const":
            return Failure()

        # we cannot elide a node if its name is in fetch, as it would not
        # be eliminated by dce pass later, which will cause name duplication
        for sig in node.func().get_total_func_sigs():
            if normalize_tensor_name(node.name) in sig.fetch:
                return Failure()

        new_name = self.use_existed_name(node.name)
        new_name = "SonyCheckFill_" + new_name

        def defer_rewrite(g):
            old_tensor = g.get_tensor_by_name(normalize_tensor_name(node.name))
            op = old_tensor.op
            dims_tensor = op.inputs[0]
            value_tensor = op.inputs[1]
            shape = []
            value = None
            targets = []
            if dims_tensor.op.type == "Const":
                targets.extend([dims_tensor, value_tensor])
                with tf.compat.v1.Session() as sess:
                    res = sess.run(targets)
                    shape.extend(res[0])
                    value = res[1]
            elif dims_tensor.op.type == "Pack":
                targets.extend(dims_tensor.op.inputs)
                targets.append(value_tensor)
                with tf.compat.v1.Session() as sess:
                    res = sess.run(targets)
                    shape.extend(res[0 : -1])
                    value = res[-1]
            else:
                return

            print(f"shape = {shape}")
            print(f"value = {value}")
            print()
            new_tensor = tf.constant(value, shape=shape, name=new_name)    
            replace(old_tensor, new_tensor)
            save_pb(f"graph.after.ElideFill.pb", g.as_graph_def(add_shapes=True))

        self.add_defer_rewrite(node.parent(), defer_rewrite)
        return Success()