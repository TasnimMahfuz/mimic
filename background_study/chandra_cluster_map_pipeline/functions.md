## File: `ParseOutput_xspec.py`

### `ParseOutput(inputdir)`

**Purpose:**  
Parses XSPEC spectral fitting results for multiple regions and writes summarized parameters to a single data file named `regions-info-xspec.data`.

**Parameters:**  
- `inputdir` — Directory path containing XSPEC result files (`reg_<n>_data.xcm`) for each region.

**Description:**  
1. Opens or creates `regions-info-xspec.data` and writes a header line with parameter names.  
2. Iterates over all regions (`range(brd.sexnum)`), attempting to open each region’s XSPEC output file.  
3. Extracts physical and statistical quantities:  
   - Hydrogen column density (`nH`)  
   - Temperature and its bounds (`temp`, `templow`, `temphigh`)  
   - Elemental abundance and its bounds (`abund`, `abundlow`, `abundhigh`)  
   - Redshift (`redshift`)  
   - Normalization and its errors (`norm`, `normlow`, `normhigh`)  
   - Fit quality (`chi`, `dof`, and computed `chi2 = chi / dof`)  
4. Computes derived errors and ranges (e.g., `temp_error_low`, `temp_error_high`, `temp_error_diff`) using NumPy.  
5. Handles missing files gracefully using `try/except` blocks (a modification by Zareef).  
6. Writes one formatted data line per region into the output file.  

**Output:**  
A text file `regions-info-xspec.data` with columns:
region nH temperature temp_low temp_high temp_error_low temp_error_high temp_error_diff abundance ...


**Dependencies:**  
- `bin_region_directories` as `brd` — provides region count (`sexnum`) and result directory (`resultsdir`)  
- `numpy` — used for numeric calculations and rounding  

**Notes:**  
- The function is called at the end of the script as `ParseOutput(brd.resultsdir)`.  
- Originates from the Chandra pipeline (`pipeline.py` by jpbreuer`), later modified by Zareef to improve robustness for missing data.  

### Limitations / Observations

- **Too many repeated variables:**  
  The function defines many individual variables (`nh`, `temp`, `templow`, etc.) at the top and then reassigns them repeatedly inside the loop. In the `except` block, these variables are redundantly reassigned (`nh = nh`, etc.).  
  *Improvement:* Use a **dictionary or data class** to hold all parameters for each region. This makes the code cleaner, avoids repetitive assignments, and improves readability.

  **Example:**
  ```python
  region_data = {
      "nh": "", "temp": "", "templow": "", "temphigh": "", ...
  }

  try:
      # parse values
      region_data["nh"] = info[0]
      region_data["temp"] = info[1]
      ...
  except:
      # no need to reassign, just keep previous values
      pass

- **Monolithic function:**  
  `ParseOutput` is very long and performs multiple responsibilities: opening/writing the output file, looping over regions, reading input files, computing derived values, and formatting output.  
  This violates the **Single Responsibility Principle**.  

  *Improvement:* Split into smaller, focused functions, for example:
  ```python
  def read_region_file(filepath):
      ...

  def compute_errors(info):
      ...

  def write_region_line(file, region_index, data):
      ...
- **Hardcoded output filename:**  
  The output file `'regions-info-xspec.data'` is hardcoded inside the function.  

  *Improvement:* Pass the filename as an argument:
  ```python
  def ParseOutput(inputdir, output_file="regions-info-xspec.data"):
      ...

- **Error handling:**
Broad except: blocks catch all exceptions and may silently hide bugs. It is recommended to catch specific exceptions `(e.g., FileNotFoundError, IndexError)` and provide warnings or logging.

```python
    except FileNotFoundError:
        print(f"Warning: File {filepath} missing, using previous values.")
```

### Suggested Design Pattern

The current implementation of `ParseOutput_xspec.py` can be refactored to follow a clearer and more maintainable structure using the **Template Method** pattern, with the option to extend it later using the **Strategy** pattern.

---

#### Template Method Pattern

**Why:**  
The overall workflow — looping over regions, reading data, computing derived values, and writing output — is fixed, but each step can be made modular and overrideable.  
This aligns with the **Single Responsibility Principle** and improves clarity, testability, and future extensibility.

**How:**  
Define a base `Parser` class that contains the general algorithm skeleton (`parse()`), while allowing subclasses to override individual steps like data reading and computation.

```python
class BaseParser:
    def parse(self, input_dir, output_file):
        regions = self.get_regions(input_dir)
        with open(output_file, 'w') as f:
            self.write_header(f)
            for region_path in regions:
                info = self.read_region(region_path)
                data = self.compute_errors(info)
                self.write_region(f, data)

    def get_regions(self, input_dir):
        raise NotImplementedError

    def read_region(self, path):
        raise NotImplementedError

    def compute_errors(self, info):
        raise NotImplementedError

    def write_region(self, file, data):
        raise NotImplementedError
```

Then implement a subclass for the XSPEC data format:

```python
class XspecParser(BaseParser):
    def get_regions(self, input_dir):
        return [f"{input_dir}/xspec/reg_{i}_data.xcm" for i in range(brd.sexnum)]

    def read_region(self, path):
        with open(path) as f:
            return f.read().split()

    def compute_errors(self, info):
        # Same computation logic as in the current implementation
        ...

    def write_region(self, file, data):
        file.write(" ".join(data) + "\n")
```
This structure lets one easily add other parser types (for different instruments or formats) without rewriting the core workflow.

### Optional: Strategy Pattern

If we later need multiple parsing “strategies” (e.g., XSPEC vs. another instrument), you can define separate computation or reading strategies and inject them into a generic parser.
This separates how data is processed from how the parsing is orchestrated, improving flexibility.
