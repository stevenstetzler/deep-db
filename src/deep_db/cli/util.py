import click


class MultiValueOption(click.Option):
    """
    A click.Option that accepts one or more space-separated values per
    occurrence of the flag (like argparse's nargs='+'), on top of the usual
    multiple=True repetition -- so `--detnum 5 6` and `--detnum 5 --detnum 6`
    both work and produce the same flattened tuple. Requires multiple=True.

    click.Option has no built-in way to do this: nargs is fixed per-option
    (and nargs=-1 is rejected outright for options, only allowed for
    arguments), so a bare nargs='+' isn't available. This works around that
    by hooking the low-level parser to greedily consume tokens up to the
    next flag, then flattening/convering the result in type_cast_value
    (click's own multiple+nargs handling expects one tuple of scalars per
    occurrence, not the list-of-tuples this produces).
    """

    def __init__(self, *args, **kwargs):
        if not kwargs.get("multiple"):
            raise TypeError("MultiValueOption requires multiple=True")
        super().__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # grab every following token up to the next --flag/-f
            value = [value]
            while state.rargs and not any(
                state.rargs[0].startswith(prefix) for prefix in self._eat_all_parser.prefixes
            ):
                value.append(state.rargs.pop(0))
            self._previous_parser_process(tuple(value), state)

        retval = super().add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval

    def type_cast_value(self, ctx, value):
        if value is None:
            return ()
        flat = [item for group in value for item in group]
        return tuple(self.type(item, self, ctx) for item in flat)
