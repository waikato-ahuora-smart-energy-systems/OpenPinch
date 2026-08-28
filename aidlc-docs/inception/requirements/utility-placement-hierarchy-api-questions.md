# Utility Placement Hierarchy API Questions

The placement API must identify both the hierarchy level that owns the
utilities and the target profile used at that level. Please answer each
question by placing a letter after its `[Answer]:` tag.

## Question 1
How should the public API make the utility ownership level explicit?

A) Use separate discoverable methods: `process_utility_placement(...)`,
`site_utility_placement(...)`, and `community_utility_placement(...)`.

B) Use one grouped namespace:
`problem.target.utility_placement.process(...)`, `.site(...)`, and
`.community(...)`.

C) Keep one method but require a typed level:
`utility_placement(level=ZoneType.S, ...)`.

D) Keep one method but require a zone and infer its level from the zone type:
`utility_placement(zone="Site/Plant", ...)`.

E) Other (please describe after the [Answer]: tag below)

[Answer]: By default, it should apply to whatever type the master_zone is within PinchProblem. One should also be able to specify a zone name/obj.

## Question 2
Which target profile should each hierarchy-level method optimize against?

A) Process uses its direct GCC; Site uses its Total Site Profile; Community and
Region use their indirect aggregate utility profiles.

B) Every hierarchy level may choose direct or indirect targeting through a
second explicit method or selector.

C) Limit this release to Process direct GCC and Site Total Site Profile;
Community and Region placement remain unsupported until separately designed.

D) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 3
How explicit must zone selection be within the chosen hierarchy level?

A) Always require a zone name or `Zone` object, even when the problem root is
the only matching zone.

B) Permit an omitted zone only when the problem root has the level named by the
method; otherwise require a zone name or `Zone` object.

C) Automatically select the only matching zone anywhere in the hierarchy and
require a zone only when multiple matches exist.

D) Other (please describe after the [Answer]: tag below)

[Answer]: C

## Question 4
What should the single executable notebook demonstrate after this API change?

A) One Process placement followed by its standard GCC, and one Site placement
followed by its standard Total Site Profile; both use two isothermal and two
sensible utility levels per side.

B) Site placement and its standard Total Site Profile only.

C) Process placement and its standard GCC only; document Site placement in RTD.

D) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 5
When the selected `PinchProblem` already contains utilities, how should explicit
level counts interact with inferred counts?

A) Existing utilities become the placement templates and counts may be omitted;
if counts are also supplied, they must match the inferred utility families.

B) Existing utilities become the placement templates and always determine the
counts; any supplied counts are ignored.

C) Counts override the existing utilities and cause new generic templates to be
generated.

D) Other (please describe after the [Answer]: tag below)

[Answer]: C

## Question 6
How should ordinary existing utilities be classified as isothermal or sensible
when they do not carry an explicit placement-kind field?

A) Infer from their thermal definition: a supply-to-target span within the
configured isothermal tolerance is isothermal; a larger span or multi-point
profile is sensible.

B) Add a required explicit `placement_kind` field to every utility used by the
optimizer.

C) Treat all ordinary utilities as isothermal and only segmented or profiled
utilities as sensible.

D) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 7
What should happen when inferred hot and cold utility sets have different
numbers of isothermal or sensible levels?

A) Preserve and optimize each side independently; inferred hot and cold counts
do not need to match.

B) Reject asymmetric utility sets and require equal hot/cold counts for each
kind, matching the current generated-template rule.

C) Pad the smaller side with generated generic templates until the counts
match.

D) Other (please describe after the [Answer]: tag below)

[Answer]: C. Each level can be either hot or cold utility.
