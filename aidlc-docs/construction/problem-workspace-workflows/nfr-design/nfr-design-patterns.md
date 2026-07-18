# NFR Design Patterns

- **Explicit command/query separation**: component/target/design methods mutate
  or execute; observation methods only read cached state.
- **Ephemeral override scope**: validated config copies are installed only for
  one invocation and restored in `finally` blocks.
- **Ordered isolation**: case batches and all-period runs retain input order and
  isolate failures/results by item.
- **Lazy adapter**: grid and figure construction import presentation extras only
  when called.
- **Closed public vocabulary**: named methods replace algorithm and graph-type
  selector strings.
