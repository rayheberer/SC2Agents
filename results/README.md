# Results
Model details, reporting on hyperparameter setings for runs and their results, and links to trained model checkpoints.

Replays and Tensorboard metrics for each individual run are stored in corresponding directories. Note that SC2 replays will not work on versions of the game other than that with which they were generated.

## Summary (best runs)
<table align="center">
  <tr>
    <td align="center"></td>
    <td align="center">MoveToBeacon</td>
    <td align="center">CollectMineralShards</td>
    <td align="center">FindAndDefeatZerglings</td>
    <td align="center">DefeatRoaches</td>
    <td align="center">DefeatZerglingsAndBanelings</td>
    <td align="center">CollectMineralsAndGas</td>
    <td align="center">BuildMarines</td>

  </tr>
  <tr>
    <td align="center">DQNMoveOnly</td>
    <td align="right">
      Mean Score: ~20<br>
      Max Score: 23
    </td>
    <td align="right">
      Mean Score: ~75<br>
      Max Score: 99
    </td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
  </tr>
  <tr>
    <td align="right">A2CAtari</td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
    <td align="right"></td>
  </tr>
</table>