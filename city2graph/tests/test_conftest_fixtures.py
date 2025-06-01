"""Tests to ensure conftest.py fixtures are properly covered."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString


def test_complex_line_fixture(complex_line: LineString) -> None:
    """Test the complex_line fixture from conftest.py."""
    assert isinstance(complex_line, LineString)
    assert len(complex_line.coords) == 5  # 5 coordinate pairs
    # Should start at (0, 0) and end at (10, 0)
    assert complex_line.coords[0] == (0, 0)
    assert complex_line.coords[-1] == (10, 0)


def test_tunnel_road_flags_fixture(tunnel_road_flags: dict[str, str]) -> None:
    """Test the tunnel_road_flags fixture from conftest.py."""
    assert isinstance(tunnel_road_flags, dict)
    assert "full" in tunnel_road_flags
    assert "partial" in tunnel_road_flags
    assert "multiple" in tunnel_road_flags
    assert "none" in tunnel_road_flags

    # Check that each value is a JSON string
    import json
    for key, value in tunnel_road_flags.items():
        try:
            parsed = json.loads(value)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON in tunnel_road_flags[{key}]: {value}")


def test_grid_data_fixture(grid_data: dict[str, gpd.GeoDataFrame]) -> None:
    """Test the grid_data fixture from conftest.py."""
    assert isinstance(grid_data, dict)
    assert "buildings" in grid_data
    assert "roads" in grid_data
    assert "tessellations" in grid_data

    # Test buildings
    buildings = grid_data["buildings"]
    assert isinstance(buildings, gpd.GeoDataFrame)
    assert len(buildings) == 9  # 3x3 grid
    assert "id" in buildings.columns
    assert buildings.crs == "EPSG:27700"

    # Test roads
    roads = grid_data["roads"]
    assert isinstance(roads, gpd.GeoDataFrame)
    assert len(roads) == 8  # 4 horizontal + 4 vertical
    assert "id" in roads.columns
    assert "subtype" in roads.columns
    assert "class" in roads.columns
    assert "road_flags" in roads.columns
    assert roads.crs == "EPSG:27700"

    # Test tessellations
    tessellations = grid_data["tessellations"]
    assert isinstance(tessellations, gpd.GeoDataFrame)
    assert len(tessellations) == 9  # 3x3 grid
    assert "tess_id" in tessellations.columns
    assert "enclosure_index" in tessellations.columns
    assert tessellations.crs == "EPSG:27700"


def test_grid_data_geometry_properties(grid_data: dict[str, gpd.GeoDataFrame]) -> None:
    """Test geometric properties of the grid_data fixture."""
    buildings = grid_data["buildings"]
    roads = grid_data["roads"]
    tessellations = grid_data["tessellations"]

    # All buildings should be squares
    for geom in buildings.geometry:
        bounds = geom.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        assert abs(width - 10) < 0.01  # 10x10 squares
        assert abs(height - 10) < 0.01

    # Roads should be LineStrings
    for geom in roads.geometry:
        assert isinstance(geom, LineString)

    # Tessellations should be slightly larger than buildings
    for geom in tessellations.geometry:
        bounds = geom.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        assert abs(width - 12) < 0.01  # 12x12 tessellations
        assert abs(height - 12) < 0.01


def test_fixture_consistency(simple_line: LineString) -> None:
    """Test that fixtures work together consistently."""
    # Test using the simple_line fixture properly
    assert isinstance(simple_line, LineString)
    assert simple_line.coords[0] == (0, 0)
    assert simple_line.coords[-1] == (10, 0)
