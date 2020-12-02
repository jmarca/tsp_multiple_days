from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from functools import partial
import argparse
import time_details as T

def main():

    parser = argparse.ArgumentParser(description='Solve routing problem to a fixed set of destinations')
    parser.add_argument('--days', type=int, dest='days', default=2,
                        help='Number of days to schedule.  Default is 2 days')
    parser.add_argument('--start', type=int, dest='start', default=6,
                        help='The earliest any trip can start on any day, in hours. Default 6')
    parser.add_argument('--end', type=int, dest='end', default=18,
                        help='The earliest any trip can end on any day, in hours.  Default 18 (which is 6pm)')
    parser.add_argument('--waittime', type=int, dest='service', default=30,
                        help='The time required to wait at each visited node, in minutes.  Default is 30')
    parser.add_argument('-t, --timelimit', type=int, dest='timelimit', default=10,
                        help='Maximum run time for solver, in seconds.  Default is 10 seconds.')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help="Turn on solver logging.")
    parser.add_argument('--guided_local', action='store_true', dest='guided_local',
                        default=False,
                        help='whether or not to use the guided local search metaheuristic')

    args = parser.parse_args()
    day_start = args.start * 3600
    day_end = args.end * 3600

    if args.days <= 0:
        print("--days parameter must be 1 or more")
        assert args.days > 0

    num_days = min(0, args.days - 1)

    node_service_time = args.service * 60
    overnight_time = -18*3600

    disjunction_penalty = 10000000

    Slack_Max = 3600 * 24 # one day
    Capacity = 3600 * 24 # one day

    num_nodes = T.num_nodes()
    # create dummy nodes for returning to the depot every night
    night_nodes = range(num_nodes, num_nodes+num_days)

    # create dummy nodes linked to night nodes that fix the AM depart time
    morning_nodes = range(num_nodes+num_days, num_nodes+num_days+num_days)

    total_nodes = num_nodes + len(night_nodes) +len(morning_nodes)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(total_nodes, 1, [0], [1])

    print('made manager with total nodes {} = {} + {} + {}'.format(total_nodes,
                                                                   num_nodes,
                                                                   len(night_nodes),
                                                                   len(morning_nodes)))
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_fn = partial(T.transit_callback,
                                  manager,
                                  day_end,
                                  night_nodes,
                                  morning_nodes)

    transit_callback_index = routing.RegisterTransitCallback(transit_callback_fn)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print('set the arc cost evaluator for all vehicles')

    time_callback_fn = partial(T.time_callback,
                               manager,
                               node_service_time,
                               overnight_time,
                               night_nodes,
                               morning_nodes)

    time_callback_index = routing.RegisterTransitCallback(time_callback_fn)


    # Define cost of each arc.

    # create time dimension
    routing.AddDimension(
        time_callback_index,
        Slack_Max,  # An upper bound for slack (the wait times at the locations).
        Capacity,  # An upper bound for the total time over each vehicle's route.
        False,  # Determine whether the cumulative variable is set to zero at the start of the vehicle's route.
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    print('created time dimension')

    # get rid of slack for all regular nodes, all morning nodes
    for node in range(2, num_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.SlackVar(index).SetValue(0)
    for node in morning_nodes:
      index = manager.NodeToIndex(node)
      time_dimension.SlackVar(index).SetValue(0)


    # Allow all locations except the first two to be droppable.
    for node in range(2, num_nodes):
      routing.AddDisjunction([manager.NodeToIndex(node)], disjunction_penalty)

    # Allow all overnight nodes to be dropped for free
    for node in range(num_nodes, total_nodes):
      routing.AddDisjunction([manager.NodeToIndex(node)], 0)

    # Add time window constraints for each regular node
    for node in range(2,num_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_end)

    # This also applies to the overnight nodes and morning nodes
    for node in range(num_nodes, total_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_end)


    # Add time window constraints for each vehicle start node.
    for veh in range(0,1):
      index = routing.Start(veh)
      time_dimension.CumulVar(index).SetMin(day_start)
      index = routing.End(veh)
      time_dimension.CumulVar(index).SetMax(day_end)

    print('done with time constraints')

    # make sure the days happen in order. first end day 1, end day 2, etc, then node 1
    # create counting dimension
    routing.AddConstantDimension(1,  # increment by 1
                                 total_nodes+1, # the max count is visit every node
                                 True, # start count at zero
                                 "Counting")
    count_dimension = routing.GetDimensionOrDie('Counting')

    print('created count dim')

    # use count dim to enforce ordering of overnight, morning nodes
    solver = routing.solver()
    for i in range(len(night_nodes)):
      inode = night_nodes[i]
      iidx = manager.NodeToIndex(inode)
      iactive = routing.ActiveVar(iidx)

      for j in range(i+1, len(night_nodes)):
        # make i come before j using count dimension
        jnode = night_nodes[j]
        jidx = manager.NodeToIndex(jnode)
        jactive = routing.ActiveVar(jidx)

        solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                   count_dimension.CumulVar(jidx) * iactive * jactive)

    for i in range(len(morning_nodes)):
      inode = morning_nodes[i]
      iidx = manager.NodeToIndex(inode)
      iactive = routing.ActiveVar(iidx)

      for j in range(i+1, len(morning_nodes)):
        # make i come before j using count dimension
        jnode = morning_nodes[j]
        jidx = manager.NodeToIndex(jnode)
        jactive = routing.ActiveVar(jidx)

        solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                   count_dimension.CumulVar(jidx) * iactive * jactive)





    # link overnight, morning nodes
    # solver = routing.solver()
    # for (night_node, morning_node) in zip(night_nodes, morning_nodes):
    #   night_index = manager.NodeToIndex(night_node)
    #   # can only go from night to morning
    #   for other in range(total_nodes):
    #     if other in [night_node, morning_node]:
    #       continue;
    #     other_index = manager.NodeToIndex(other)
    #     routing.NextVar(night_index).RemoveValue(other_index)
    # print('done setting up ordering constraints between days')

    # for (night_node, morning_node) in zip(night_nodes, morning_nodes):
    #   # print('did not die', night_node, morning_node)
    #   morning_index = manager.NodeToIndex(morning_node)
    #   for other in range(total_nodes):
    #     if other in [0, 1, night_node, morning_node]:
    #       continue;
    #     # print('did not die', other)
    #     other_index = manager.NodeToIndex(other)
    #     if routing.NextVar(other_index).Contains(morning_index):
    #       routing.NextVar(other_index).RemoveValue(morning_index)
    # #assert 0

      # constraint = routing.NextVar(night_index) == morning_index
      # # this node comes after prior day
      # active_night = routing.ActiveVar(night_index)
      # active_morning = routing.ActiveVar(morning_index)
      # solver.Add(active_night == active_morning)
      # conditional_expr = solver.ConditionalExpression(active_night,
      #                                                 constraint,
      #                                                 1)
      # solver.Add(conditional_expr >= 1)
      # prior_index = index

    print('done setting up ordering constraints between days')

    # Instantiate route start and end times to produce feasible times.
    # routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(0)))
    # routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))


    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    print('set up the setup.  Total nodes is ', total_nodes, ' and real nodes is ', num_nodes)
    # Setting local search metaheuristics:
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = args.timelimit
    search_parameters.log_search = args.debug

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
      print("no solution found")
    else:
      print("solution found.  Objective value is ",solution.ObjectiveValue())

      # Print the results
      result = {
        'Dropped': [],
        'Scheduled': []
      }

      # Return the dropped locations
      for index in range(routing.Size()):
        if routing.IsStart(index) or routing.IsEnd(index):
          continue
        node = manager.IndexToNode(index)
        if node in night_nodes or node in morning_nodes:
          continue
        if solution.Value(routing.NextVar(index)) == index:
          result['Dropped'].append(node)

      # Return the scheduled locations
      time = 0
      index = routing.Start(0)
      while not routing.IsEnd(index):
        time = time_dimension.CumulVar(index)
        count = count_dimension.CumulVar(index)
        node = manager.IndexToNode(index)
        if node in night_nodes:
          node = 'Overnight at {}, dummy for 1'.format(node)
        if node in morning_nodes:
          node = 'Starting day at {}, dummy for 1'.format(node)

        result['Scheduled'].append([node, solution.Value(count), solution.Min(time)/3600,solution.Max(time)/3600])
        index = solution.Value(routing.NextVar(index))

      time = time_dimension.CumulVar(index)
      count = count_dimension.CumulVar(index)
      result['Scheduled'].append([manager.IndexToNode(index), solution.Value(count), solution.Min(time)/3600,solution.Max(time)/3600])

      print('Dropped')
      print(result['Dropped'])

      print('Scheduled')
      for line in result['Scheduled']:
        print(line)

      #print(result)

if __name__ == '__main__':
    main()
