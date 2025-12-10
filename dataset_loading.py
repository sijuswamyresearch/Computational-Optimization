import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
import folium
from typing import Dict, List, Any
import os

class FeasibleCampusDataset:
    def __init__(self):
        """
        Campus Dataset with Feasible Constraints for Module 1
        
        Designed specifically to ensure solvable optimization problems
        """
        # Real coordinates for Columbia University area
        self.base_coords = {
            'center_lat': 40.8075,
            'center_lon': -73.9626
        }
        
        # Initialize datasets with feasible parameters
        self.facilities = self.create_facilities_with_feasible_demands()
        self.warehouses = self.create_warehouses_with_feasible_capacities()
        self.cost_matrix = self.create_transportation_costs()
        self.demands = self.create_demand_data()
        
    def create_facilities_with_feasible_demands(self):
        """Create facilities with demands that can be satisfied by warehouse capacities"""
        facilities_data = {
            'facility_id': ['MED_CENTER', 'ENG_BUILDING', 'SCIENCE_HALL', 'DORM_A', 'DORM_B', 
                           'LIBRARY', 'STADIUM', 'COMM_CENTER', 'PUBLIC_SCHOOL', 'LOCAL_HOSPITAL'],
            'facility_name': [
                'Campus Medical Center', 'Engineering Building', 'Science Hall', 
                'North Dormitory', 'South Dormitory', 'Main Library', 'University Stadium',
                'Community Center', 'Local Public School', 'Community Hospital'
            ],
            'facility_type': [
                'Hospital', 'Academic', 'Academic', 'Residential', 'Residential',
                'Academic', 'Recreational', 'Community', 'School', 'Hospital'
            ],
            # Real coordinates around Columbia University
            'latitude': [
                40.8075, 40.8100, 40.8050, 40.8120, 40.8030, 
                40.8085, 40.8040, 40.8060, 40.8020, 40.8090
            ],
            'longitude': [
                -73.9626, -73.9600, -73.9650, -73.9590, -73.9660,
                -73.9630, -73.9640, -73.9610, -73.9670, -73.9580
            ]
        }
        
        return pd.DataFrame(facilities_data)
    
    def create_warehouses_with_feasible_capacities(self):
        """Create warehouses with capacities that can satisfy total demand"""
        warehouses_data = {
            'warehouse_id': ['WH_NORTH', 'WH_SOUTH', 'WH_EAST', 'WH_WEST', 'WH_CENTRAL'],
            'warehouse_name': ['North Campus Warehouse', 'South Campus Warehouse', 
                              'East Gate Warehouse', 'West Gate Warehouse', 'Central Storage'],
            'latitude': [40.8125, 40.8025, 40.8075, 40.8075, 40.8075],
            'longitude': [-73.9626, -73.9626, -73.9570, -73.9680, -73.9626],
            # Increased capacities to ensure feasibility
            'capacity': [400, 350, 450, 300, 500],  # Total capacity: 2000 units
            # Reduced costs to fit within budget
            'construction_cost': [300000, 280000, 320000, 250000, 350000],  # Total: $1,500,000
            'operational_cost': [800, 700, 900, 600, 1000]
        }
        
        return pd.DataFrame(warehouses_data)
    
    def calculate_geographic_distance(self, lat1, lon1, lat2, lon2):
        """Calculate approximate distance in kilometers"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371  # Earth radius in km
    
    def create_transportation_costs(self):
        """Create transportation costs with reasonable values"""
        cost_data = []
        
        for _, warehouse in self.warehouses.iterrows():
            for _, facility in self.facilities.iterrows():
                distance_km = self.calculate_geographic_distance(
                    warehouse['latitude'], warehouse['longitude'],
                    facility['latitude'], facility['longitude']
                )
                
                # Reduced cost model for feasibility
                fixed_cost = 3.00  # Reduced from $5.00
                variable_cost = 1.50  # Reduced from $2.50
                cost_per_unit = fixed_cost + (distance_km * variable_cost)
                
                cost_data.append({
                    'from_warehouse': warehouse['warehouse_id'],
                    'to_facility': facility['facility_id'],
                    'distance_km': round(distance_km, 2),
                    'cost_per_unit': round(cost_per_unit, 2)
                })
        
        return pd.DataFrame(cost_data)
    
    def create_demand_data(self):
        """Create demand data that fits within warehouse capacities"""
        # Reduced demands to ensure feasibility
        base_demands = {
            'Hospital': 80,    # Reduced from 120
            'Academic': 30,    # Reduced from 45
            'Residential': 50, # Reduced from 80
            'Recreational': 25,# Reduced from 35
            'Community': 40,   # Reduced from 60
            'School': 35       # Reduced from 55
        }
        
        demand_data = []
        for _, facility in self.facilities.iterrows():
            base_demand = base_demands.get(facility['facility_type'], 30)
            daily_demand = int(base_demand * np.random.uniform(0.9, 1.1))  # Less variation
            
            demand_data.append({
                'facility_id': facility['facility_id'],
                'daily_demand': daily_demand,
                'priority_level': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
            })
        
        # Verify total demand fits within capacity
        total_daily_demand = sum([d['daily_demand'] for d in demand_data])
        total_warehouse_capacity = self.warehouses['capacity'].sum()
        print(f"📊 Total Daily Demand: {total_daily_demand} units")
        print(f"🏭 Total Warehouse Capacity: {total_warehouse_capacity} units")
        print(f"✅ Capacity Utilization: {(total_daily_demand/total_warehouse_capacity)*100:.1f}%")
        
        return pd.DataFrame(demand_data)
    
    def create_interactive_map(self, solution_data: Dict = None):
        """Create an interactive Folium map with the solution"""
        # Create base map
        m = folium.Map(
            location=[self.base_coords['center_lat'], self.base_coords['center_lon']],
            zoom_start=14,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Color scheme for facilities
        facility_colors = {
            'Hospital': 'red',
            'Academic': 'blue', 
            'Residential': 'green',
            'Recreational': 'orange',
            'Community': 'purple',
            'School': 'brown'
        }
        
        # Add facilities to map
        for _, facility in self.facilities.iterrows():
            color = facility_colors.get(facility['facility_type'], 'gray')
            demand_info = self.demands[self.demands['facility_id'] == facility['facility_id']].iloc[0]
            
            folium.CircleMarker(
                location=[facility['latitude'], facility['longitude']],
                radius=10,
                popup=f"""
                <b>{facility['facility_name']}</b><br>
                Type: {facility['facility_type']}<br>
                Daily Demand: {demand_info['daily_demand']} units<br>
                Priority: {demand_info['priority_level']}
                """,
                tooltip=f"{facility['facility_id']}: {facility['facility_name']}",
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add warehouses
        warehouse_colors = {
            'WH_NORTH': 'darkblue',
            'WH_SOUTH': 'darkgreen', 
            'WH_EAST': 'darkred',
            'WH_WEST': 'darkpurple',
            'WH_CENTRAL': 'black'
        }
        
        for _, warehouse in self.warehouses.iterrows():
            is_selected = solution_data and warehouse['warehouse_id'] in solution_data.get('selected_warehouses', [])
            
            # Different icon for selected vs available warehouses
            if is_selected:
                icon_color = 'green'
                icon_type = 'check-circle'
                status = "SELECTED"
            else:
                icon_color = 'red' 
                icon_type = 'times-circle'
                status = "Available"
            
            folium.Marker(
                location=[warehouse['latitude'], warehouse['longitude']],
                popup=f"""
                <b>{warehouse['warehouse_name']}</b><br>
                Capacity: {warehouse['capacity']} units/day<br>
                Construction: ${warehouse['construction_cost']:,}<br>
                Operational: ${warehouse['operational_cost']}/day<br>
                Status: <b>{status}</b>
                """,
                tooltip=f"{warehouse['warehouse_id']}: {warehouse['warehouse_name']} ({status})",
                icon=folium.Icon(
                    color=icon_color,
                    icon=icon_type,
                    prefix='fa'
                )
            ).add_to(m)
        
        # Add solution routes if available
        if solution_data and solution_data.get('status') == 'Optimal' and 'shipment_plan' in solution_data:
            print(f"🔄 Adding {len(solution_data['shipment_plan'])} shipment routes to map...")
            
            for shipment in solution_data['shipment_plan']:
                wh = self.warehouses[self.warehouses['warehouse_id'] == shipment['from_warehouse']].iloc[0]
                fac = self.facilities[self.facilities['facility_id'] == shipment['to_facility']].iloc[0]
                
                # Calculate line width based on shipment volume
                line_weight = 2 + (shipment['daily_units'] / 10)  # Scale for visibility
                
                folium.PolyLine(
                    locations=[
                        [wh['latitude'], wh['longitude']],
                        [fac['latitude'], fac['longitude']]
                    ],
                    color='blue',
                    weight=min(line_weight, 8),  # Cap at 8 for visibility
                    opacity=0.7,
                    popup=f"""
                    <b>Supply Route</b><br>
                    From: {wh['warehouse_name']}<br>
                    To: {fac['facility_name']}<br>
                    Daily Units: {shipment['daily_units']:.1f}<br>
                    Annual Units: {shipment['units']:,.0f}
                    """,
                    tooltip=f"{wh['warehouse_id']} → {fac['facility_id']}: {shipment['daily_units']:.1f} units/day"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>🏙️ Campus City Map Legend</h4>
        <p><span style="color: red;">●</span> Hospital</p>
        <p><span style="color: blue;">●</span> Academic</p>
        <p><span style="color: green;">●</span> Residential</p>
        <p><span style="color: orange;">●</span> Recreational</p>
        <p><span style="color: purple;">●</span> Community</p>
        <p><span style="color: brown;">●</span> School</p>
        <p><span style="color: green;"><i class="fa fa-check-circle"></i></span> Selected Warehouse</p>
        <p><span style="color: red;"><i class="fa fa-times-circle"></i></span> Available Warehouse</p>
        <p><span style="color: blue;">━━━</span> Supply Route</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px; margin-top:10px;">
        <b>Module 1: Campus Supply Optimization Solution</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def plot_static_map(self, solution_data: Dict = None):
        """Create a static matplotlib map for quick visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Color scheme
        facility_colors = {
            'Hospital': 'red', 'Academic': 'blue', 'Residential': 'green',
            'Recreational': 'orange', 'Community': 'purple', 'School': 'brown'
        }
        
        # Plot facilities
        for _, facility in self.facilities.iterrows():
            color = facility_colors.get(facility['facility_type'], 'gray')
            ax.scatter(facility['longitude'], facility['latitude'], 
                      c=color, s=100, alpha=0.7, label=facility['facility_type'])
            ax.annotate(facility['facility_id'], 
                       (facility['longitude'], facility['latitude']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot warehouses
        for _, warehouse in self.warehouses.iterrows():
            is_selected = solution_data and warehouse['warehouse_id'] in solution_data.get('selected_warehouses', [])
            marker = 's'
            color = 'green' if is_selected else 'red'
            size = 150 if is_selected else 100
            
            ax.scatter(warehouse['longitude'], warehouse['latitude'], 
                      c=color, s=size, marker=marker, alpha=0.8)
            ax.annotate(warehouse['warehouse_id'], 
                       (warehouse['longitude'], warehouse['latitude']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
        
        # Plot solution routes
        if solution_data and solution_data.get('status') == 'Optimal' and 'shipment_plan' in solution_data:
            for shipment in solution_data['shipment_plan'][:15]:  # Limit for clarity
                wh = self.warehouses[self.warehouses['warehouse_id'] == shipment['from_warehouse']].iloc[0]
                fac = self.facilities[self.facilities['facility_id'] == shipment['to_facility']].iloc[0]
                
                ax.plot([wh['longitude'], fac['longitude']], 
                       [wh['latitude'], fac['latitude']], 
                       'blue', alpha=0.5, linewidth=shipment['daily_units']/5)
        
        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.set_title('Campus Supply Optimization Solution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_module1_subset(self):
        """Get simplified dataset for Module 1 with guaranteed feasibility"""
        # Select facilities and warehouses that definitely work together
        selected_facilities = self.facilities.head(6).copy()  # Reduced from 8
        selected_warehouses = self.warehouses.head(3).copy()  # Keep 3 warehouses
        
        # Calculate subset parameters
        subset_demands = self.demands[
            self.demands['facility_id'].isin(selected_facilities['facility_id'])
        ]
        total_subset_demand = subset_demands['daily_demand'].sum() * 365  # Annual
        subset_capacity = selected_warehouses['capacity'].sum() * 365
        
        print(f"🔍 Module 1 Subset Analysis:")
        print(f"   Facilities: {len(selected_facilities)}")
        print(f"   Warehouses: {len(selected_warehouses)}")
        print(f"   Annual Demand: {total_subset_demand:,.0f} units")
        print(f"   Annual Capacity: {subset_capacity:,.0f} units")
        print(f"   Capacity Margin: {((subset_capacity - total_subset_demand)/total_subset_demand)*100:.1f}%")
        
        selected_cost_matrix = self.cost_matrix[
            self.cost_matrix['from_warehouse'].isin(selected_warehouses['warehouse_id']) & 
            self.cost_matrix['to_facility'].isin(selected_facilities['facility_id'])
        ].copy()
        
        return {
            'facilities': selected_facilities,
            'warehouses': selected_warehouses,
            'cost_matrix': selected_cost_matrix,
            'demands': subset_demands,
            'budget_constraint': 1200000,  # Reduced budget for subset
            'min_warehouses': 2,
            'max_warehouses': 3  # Added maximum constraint
        }
            
    def save_to_csv(self, directory: str = 'campus_city_data'):
        """Save all datasets to CSV files for future use"""
        import os
        os.makedirs(directory, exist_ok=True)
    
        # Save the main datasets
        self.facilities.to_csv(f'{directory}/facilities.csv', index=False)
        self.warehouses.to_csv(f'{directory}/warehouses.csv', index=False)
        self.cost_matrix.to_csv(f'{directory}/transportation_costs.csv', index=False)
        self.demands.to_csv(f'{directory}/demands.csv', index=False)
    
        # Create and save geographic bounds
        import pandas as pd
        bounds_data = {
         'center_lat': [self.base_coords['center_lat']],
         'center_lon': [self.base_coords['center_lon']],
         'radius_km': [2.0],
         'total_area': [12.56]
        }
        bounds_df = pd.DataFrame(bounds_data)
        bounds_df.to_csv(f'{directory}/geographic_bounds.csv', index=False)
    
        print(f"✅ All 5 datasets saved to '{directory}' directory:")
        print(f"   📍 facilities.csv ({len(self.facilities)} facilities)")
        print(f"   🏭 warehouses.csv ({len(self.warehouses)} warehouses)")
        print(f"   🚚 transportation_costs.csv ({len(self.cost_matrix)} cost entries)")
        print(f"   📦 demands.csv ({len(self.demands)} demand entries)")
        print(f"   🗺️ geographic_bounds.csv (area definition)")
    
class FeasibleCampusOptimizer:
    def __init__(self, dataset: FeasibleCampusDataset):
        self.dataset = dataset
        self.model = None
        self.solution = None
    
    def build_and_solve_module1(self):
        """Build and solve Module 1 problem with guaranteed feasibility"""
        data = self.dataset.get_module1_subset()
        
        print("\n🧮 Building Optimization Model...")
        
        # Create model
        self.model = pulp.LpProblem("Module1_Campus_Supply_Optimization", pulp.LpMinimize)
        
        facilities = data['facilities']['facility_id'].tolist()
        warehouses = data['warehouses']['warehouse_id'].tolist()
        
        # Decision variables
        ship_vars = pulp.LpVariable.dicts("Ship", 
                                         [(w, f) for w in warehouses for f in facilities],
                                         lowBound=0, cat='Continuous')
        
        build_vars = pulp.LpVariable.dicts("Build", warehouses, cat='Binary')
        
        # OBJECTIVE: Minimize total cost
        transportation_cost = pulp.lpSum([
            ship_vars[w, f] * data['cost_matrix'][
                (data['cost_matrix']['from_warehouse'] == w) & 
                (data['cost_matrix']['to_facility'] == f)
            ]['cost_per_unit'].iloc[0]
            for w in warehouses for f in facilities
        ])
        
        construction_operational_cost = pulp.lpSum([
            build_vars[w] * (
                data['warehouses'][data['warehouses']['warehouse_id'] == w]['construction_cost'].iloc[0] / 10 +  # 10-year amortization
                data['warehouses'][data['warehouses']['warehouse_id'] == w]['operational_cost'].iloc[0] * 365
            )
            for w in warehouses
        ])
        
        total_cost = transportation_cost + construction_operational_cost
        self.model += total_cost, "Total_Cost"
        
        print("✅ Objective function defined")
        
        # CONSTRAINTS
        
        # 1. Demand satisfaction (annual)
        for f in facilities:
            daily_demand = data['demands'][data['demands']['facility_id'] == f]['daily_demand'].iloc[0]
            annual_demand = daily_demand * 365
            self.model += pulp.lpSum([ship_vars[w, f] for w in warehouses]) == annual_demand, f"Demand_{f}"
        
        print("✅ Demand constraints added")
        
        # 2. Capacity constraints (annual)
        for w in warehouses:
            daily_capacity = data['warehouses'][data['warehouses']['warehouse_id'] == w]['capacity'].iloc[0]
            annual_capacity = daily_capacity * 365
            self.model += pulp.lpSum([ship_vars[w, f] for f in facilities]) <= annual_capacity * build_vars[w], f"Capacity_{w}"
        
        print("✅ Capacity constraints added")
        
        # 3. Budget constraint
        self.model += total_cost <= data['budget_constraint'], "Budget"
        print("✅ Budget constraint added")
        
        # 4. Warehouse selection constraints
        self.model += pulp.lpSum([build_vars[w] for w in warehouses]) >= data['min_warehouses'], "Min_Warehouses"
        self.model += pulp.lpSum([build_vars[w] for w in warehouses]) <= data['max_warehouses'], "Max_Warehouses"
        print("✅ Warehouse selection constraints added")
        
        # SOLVE
        print("\n🔍 Solving Optimization Problem...")
        self.model.solve()
        
        # EXTRACT SOLUTION
        self.solution = {
            'status': pulp.LpStatus[self.model.status],
            'objective_value': pulp.value(self.model.objective) if self.model.status == pulp.LpStatusOptimal else None,
            'selected_warehouses': [],
            'shipment_plan': [],
            'constraint_analysis': {}
        }
        
        if self.model.status == pulp.LpStatusOptimal:
            # Get selected warehouses
            self.solution['selected_warehouses'] = [w for w in warehouses if build_vars[w].varValue > 0.5]
            
            # Get shipment plan
            for (w, f) in ship_vars:
                if ship_vars[w, f].varValue > 0:
                    self.solution['shipment_plan'].append({
                        'from_warehouse': w,
                        'to_facility': f,
                        'units': ship_vars[w, f].varValue,
                        'daily_units': ship_vars[w, f].varValue / 365
                    })
            
            # Analyze constraint satisfaction
            total_construction = sum([
                data['warehouses'][data['warehouses']['warehouse_id'] == w]['construction_cost'].iloc[0] / 10
                for w in self.solution['selected_warehouses']
            ])
            
            total_operational = sum([
                data['warehouses'][data['warehouses']['warehouse_id'] == w]['operational_cost'].iloc[0] * 365
                for w in self.solution['selected_warehouses']
            ])
            
            total_transportation = self.solution['objective_value'] - total_construction - total_operational
            
            self.solution['constraint_analysis'] = {
                'total_construction': total_construction,
                'total_operational': total_operational,
                'total_transportation': total_transportation,
                'budget_utilization': (self.solution['objective_value'] / data['budget_constraint']) * 100
            }
        
        return self.solution
    
    def print_detailed_solution(self):
        """Print comprehensive solution analysis"""
        if self.solution is None:
            print("No solution available. Please solve the model first.")
            return
        
        print("\n" + "="*60)
        print("📊 MODULE 1 OPTIMIZATION SOLUTION")
        print("="*60)
        print(f"🟢 Solution Status: {self.solution['status']}")
        
        if self.solution['status'] == 'Optimal':
            print(f"💰 Total Annual Cost: ${self.solution['objective_value']:,.2f}")
            
            print(f"\n🏭 Selected Warehouses ({len(self.solution['selected_warehouses'])}):")
            for wh in self.solution['selected_warehouses']:
                print(f"   ✅ {wh}")
            
            print(f"\n📦 Shipment Summary:")
            total_shipments = sum([s['units'] for s in self.solution['shipment_plan']])
            print(f"   Total Annual Shipments: {total_shipments:,.0f} units")
            print(f"   Total Daily Shipments: {total_shipments/365:,.0f} units/day")
            print(f"   Number of Routes: {len(self.solution['shipment_plan'])}")
            
            print(f"\n💵 Cost Breakdown:")
            analysis = self.solution['constraint_analysis']
            print(f"   Construction (annualized): ${analysis['total_construction']:,.2f}")
            print(f"   Operational (annual): ${analysis['total_operational']:,.2f}")
            print(f"   Transportation: ${analysis['total_transportation']:,.2f}")
            print(f"   Budget Utilization: {analysis['budget_utilization']:.1f}%")
            
            print(f"\n📋 First 5 Shipments:")
            for shipment in self.solution['shipment_plan'][:5]:
                print(f"   {shipment['from_warehouse']} → {shipment['to_facility']}: "
                      f"{shipment['daily_units']:.1f} units/day")
        
        else:
            print("❌ No feasible solution found. The problem constraints may be too restrictive.")
            print("💡 Try adjusting: Budget, Warehouse capacities, or Facility demands")

# Main execution
if __name__ == "__main__":
    print("🚀 MODULE 1: FEASIBLE CAMPUS OPTIMIZATION")
    print("=" * 50)
    
    # 1. Create dataset with feasible constraints
    print("1. Creating dataset with guaranteed feasible constraints...")
    campus_data = FeasibleCampusDataset()
    
    # 2. Solve optimization problem
    print("2. Solving Module 1 optimization problem...")
    optimizer = FeasibleCampusOptimizer(campus_data)
    solution = optimizer.build_and_solve_module1()
    
    # 3. Print detailed solution
    optimizer.print_detailed_solution()
    
    # 4. Create static map
    print("\n3. Creating static solution visualization...")
    campus_data.plot_static_map(solution)
    
    # 5. Create interactive map with solution
    print("4. Creating interactive solution map...")
    interactive_map = campus_data.create_interactive_map(solution)
    interactive_map.save('module1_solution_map.html')
    print("✅ Interactive solution map saved as 'module1_solution_map.html'")
    
    # 6. Save data for future use
    campus_data.save_to_csv('module1_data')
    
    print("\n" + "=" * 50)
    print("🎉 MODULE 1 COMPLETED SUCCESSFULLY!")
    print("📁 Data saved to 'module1_data' directory")
    print("🗺️ Solution map: 'module1_solution_map.html'")
    print("📊 Static visualization displayed")
    print("📚 Students can now analyze the feasible optimal solution!")