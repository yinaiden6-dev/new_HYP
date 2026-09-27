# New-group property restoration: frozen external and internal propagation

The locked nine groups were absent from prior F71 mechanism work but were already used in historical H593 evaluation. This is group-isolated mechanism replication, not untouched external confirmation.

Only the two preselected target/fixed-wrong M values change. Both posterior L values are recomputed under the SAME frozen POST_REAL. If RAW winner is neither pair, its original HR1 M/L stay fixed; if in the pair, its own intervened values are used. HOLD logit is exactly zero.

No training, no RoMa/vision/LLM forward, no new parameters. Only cached POST hidden states, frozen adapter/projector and reference tokens are used.

| Contrast | log M gap | NATIVE7 action gap | M_FREE action gap | POST content gap | POST action gap |
|---|---|---|---|---|---|
| phase_restoration_after_amplitude_permutation | +0.02417799 [0.007617792999211646, 0.04045021734759744] (8/9 positive groups) | +0.01708702 [0.003292857259878564, 0.03270171113132538] (8/9 positive groups) | +0.02532673 [0.005352223471499461, 0.05051564156640008] (8/9 positive groups) | +0.00217926 [0.0008167671644239744, 0.003567438543077199] (8/9 positive groups) | +0.11582082 [0.04180531990408687, 0.20136731199098393] (7/9 positive groups) |
| amplitude_restoration_under_local_phase | +0.01747705 [-0.09248771563503623, 0.12612330217444473] (4/9 positive groups) | -0.02437977 [-0.11823478178846729, 0.04868773054582005] (4/9 positive groups) | -0.03613515 [-0.14226228769060775, 0.051069172929433] (4/9 positive groups) | +0.01298236 [0.00269236523460914, 0.025768679870483417] (5/9 positive groups) | +0.70077842 [0.027002610213757435, 1.6020354747705587] (5/9 positive groups) |
| factorial_interaction | -0.00657794 [-0.01801382958724814, 0.004071270310833019] (4/9 positive groups) | -0.00278562 [-0.010981836393604406, 0.00484563883781298] (4/9 positive groups) | -0.00281798 [-0.014152537150423644, 0.007268886563192293] (4/9 positive groups) | -0.00260856 [-0.0043067335978067755, -0.000986493797114034] (2/9 positive groups) | -0.13235328 [-0.2255672820905218, -0.0442774050038422] (3/9 positive groups) |
| phase_restoration_native_amplitude | +0.01760005 [0.0018978432606201407, 0.03518643805362734] (7/9 positive groups) | +0.01430140 [-0.0021473549884880347, 0.033758715412938685] (7/9 positive groups) | +0.02250876 [-0.0012608149572210488, 0.052995980808377016] (7/9 positive groups) | -0.00042930 [-0.000863033051444788, -5.346851912941689e-05] (2/9 positive groups) | -0.01653246 [-0.035736011982287645, 0.0006410374720131494] (2/9 positive groups) |

Target and fixed wrong are also reported separately to distinguish an increased target score from suppression of the wrong candidate:

| Endpoint | Contrast | Target effect | Fixed wrong effect |
|---|---|---|---|
| logM | phase_restoration_after_amplitude_permutation | +0.02849872 [0.006322861122133995, 0.048660756017856674] (6/9 positive groups) | +0.00432073 [-0.02368406984237654, 0.03267562772719488] (5/9 positive groups) |
| logM | amplitude_restoration_under_local_phase | +0.26131585 [0.14991455174755947, 0.3790532058383195] (9/9 positive groups) | +0.24383880 [0.122916884377328, 0.38322668339982324] (9/9 positive groups) |
| logM | factorial_interaction | -0.04849031 [-0.06142275254186279, -0.035052313676838795] (0/9 positive groups) | -0.04191237 [-0.05245019986442845, -0.03265958652728658] (0/9 positive groups) |
| logM | phase_restoration_native_amplitude | -0.01999159 [-0.02955172165227133, -0.011375862841234513] (0/9 positive groups) | -0.03759164 [-0.05906724379947521, -0.016641350149206827] (1/9 positive groups) |
| NATIVE7_action | phase_restoration_after_amplitude_permutation | +0.00539091 [-0.004250430417830649, 0.014558085890969415] (3/9 positive groups) | -0.01169611 [-0.026935096134495136, -0.0017664002949931855] (0/9 positive groups) |
| NATIVE7_action | amplitude_restoration_under_local_phase | -0.02156430 [-0.11412486283638448, 0.04623162188401865] (2/9 positive groups) | +0.00281548 [-0.022539654498075084, 0.028558774839355056] (3/9 positive groups) |
| NATIVE7_action | factorial_interaction | -0.00434495 [-0.011683456935340786, 0.0024464939745718678] (1/9 positive groups) | -0.00155933 [-0.004519273569452537, 0.0010557670289719706] (2/9 positive groups) |
| NATIVE7_action | phase_restoration_native_amplitude | +0.00104596 [-0.008688572725502115, 0.010139769889946236] (3/9 positive groups) | -0.01325544 [-0.030870626500233176, -0.0007432528350352105] (1/9 positive groups) |
| M_FREE_action | phase_restoration_after_amplitude_permutation | +0.00699158 [-0.004854753354518287, 0.019091948944054905] (3/9 positive groups) | -0.01833516 [-0.043247292837424865, -0.0024056877558711843] (0/9 positive groups) |
| M_FREE_action | amplitude_restoration_under_local_phase | -0.02636318 [-0.1300107803254609, 0.0486542545470415] (2/9 positive groups) | +0.00977197 [-0.027383801831274375, 0.04942579773971424] (3/9 positive groups) |
| M_FREE_action | factorial_interaction | -0.00585079 [-0.01576148240375909, 0.00260763852186117] (1/9 positive groups) | -0.00303281 [-0.007623891483178213, 0.0008975463486827638] (2/9 positive groups) |
| M_FREE_action | phase_restoration_native_amplitude | +0.00114079 [-0.009725627180473689, 0.0109773556192472] (3/9 positive groups) | -0.02136797 [-0.05051229710498307, -0.0004945535204089916] (1/9 positive groups) |
| POST_content | phase_restoration_after_amplitude_permutation | +0.00295125 [0.0014165576272492516, 0.004369050312806349] (6/9 positive groups) | +0.00077200 [4.7260756733840614e-05, 0.0017741968051634956] (7/9 positive groups) |
| POST_content | amplitude_restoration_under_local_phase | +0.02321083 [0.010609667918558222, 0.03600238077433827] (9/9 positive groups) | +0.01022848 [0.0022686003860754177, 0.02178629808588572] (9/9 positive groups) |
| POST_content | factorial_interaction | -0.00368615 [-0.005280455220893645, -0.001900031820890199] (1/9 positive groups) | -0.00107760 [-0.0022571401494160055, -0.00019112768449263833] (0/9 positive groups) |
| POST_content | phase_restoration_native_amplitude | -0.00073490 [-0.0010329233419994477, -0.0004305668912336398] (0/9 positive groups) | -0.00030560 [-0.0007331336539845847, 0.00010031023343782606] (2/9 positive groups) |
| POST_action | phase_restoration_after_amplitude_permutation | +0.08071048 [0.015857279979353982, 0.1539292023306325] (4/9 positive groups) | -0.03511034 [-0.10730740703350995, 0.002883377858407964] (2/9 positive groups) |
| POST_action | amplitude_restoration_under_local_phase | +0.49532265 [-0.019021800303754825, 1.3335396164633753] (3/9 positive groups) | -0.20545578 [-0.6635739908542985, 0.04115568088482904] (3/9 positive groups) |
| POST_action | factorial_interaction | -0.09480271 [-0.18109349337152264, -0.023625667474206087] (0/9 positive groups) | +0.03755057 [-0.0032759491033001156, 0.11624123379752067] (2/9 positive groups) |
| POST_action | phase_restoration_native_amplitude | -0.01409223 [-0.032794097640086645, 0.00227728953524896] (1/9 positive groups) | +0.00244023 [-0.0038128876811082003, 0.010519820705054429] (4/9 positive groups) |

Intervals are exploratory group bootstraps. Fixed target/wrong pair selection and the third-RAW-winner anchor do not license a full-C128 accuracy claim.
This closes same-model property-effect propagation on the new mechanism groups; it does not perform phase/amplitude-specific deletion and retraining, establish unique causality, or demonstrate external generalization.
