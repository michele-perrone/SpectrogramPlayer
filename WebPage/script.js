const audio_prefix = "audio";
const plots_prefix = "plots";
const datasets = ['speech', 'music', 'urban'];
const techniques = ['signals', 'no_phase', 'griffin_lim', 'dgl_biased', 'dgl_unbiased', 'melgan', 'hifigan', 'uniglow', 'squeezewave', 'univnet'];
const techniques_names = ['Reference', 'No-phase', 'Griffin-Lim', 'DGL (biased)', 'DGL (unbiased)', 'MelGAN', 'HiFiGAN', 'UniGlow', 'SqueezeWave', 'UnivNet'];
const n_audio_files = 5; // Change this up to 20
var curr_player_id = 0;
var current_dataset = 0;

// Ge the container where we display the results (audio or plots)
var resultsDiv = document.getElementById('results');

// Get the buttons
var audioButton = document.getElementById('audioButton');
var plotsButton = document.getElementById('plotsButton');

// Add the listeners to the buttons
audioButton.addEventListener("click", generate_audio_table);
plotsButton.addEventListener("click", generate_plots);

// Fills resultsDiv with the plots
function generate_plots()
{
   var plots = "";

   for(idx_dataset = 0; idx_dataset < datasets.length; idx_dataset++) // For each dataset
   {
      current_dataset = datasets[idx_dataset];
      plots += "<h2 class='h2 text-center'>Comparison for " + current_dataset + "</h2>";

      // Waveform comparison
      var current_plot = plots_prefix + "/" + current_dataset + ".svg";
      plots += "<img class='img-fluid border border-2 p-3 m-3' src='" + current_plot + "'>";
   }

   resultsDiv.innerHTML = plots;
}

// Fills resultsDiv with the audio comparison table
function generate_audio_table()
{
   var table = "";

   for(idx_dataset = 0; idx_dataset < datasets.length; idx_dataset++)
   {
      current_dataset = datasets[idx_dataset];
      table += "<h2 class='h2 text-center'>Comparison for " + current_dataset + "</h2>";

      table += "<table class='table table-striped table-hover'>  <thead class='font-monospace'> <tr>"; // Generate the table head
      for(idx_technique = 0; idx_technique < techniques.length; idx_technique++)
      {
         table += "<th scope='col'>" + techniques_names[idx_technique] + "</th>";
      }
      table += "</tr> </thead>";

      // Generate the body
      table += "<tbody class='table-group-divider'>";
      for(idx_audio = 0; idx_audio < n_audio_files; idx_audio++) // For each audio file
      {
         table += "<tr>";
         for(idx_technique = 0; idx_technique < techniques.length; idx_technique++) // For each technique
         {
            var current_audio = audio_prefix + "/" + current_dataset + "/" + techniques[idx_technique] + "_" + current_dataset + "/" + idx_audio + ".m4a";
            //console.log(current_audio);

            table += "<td>";

            curr_player_id++;
            table += "<audio id='" + curr_player_id + "'> <source src='" + current_audio + "' type='audio/mpeg'> </audio>";

            table += "<div class='d-grid gap-2 d-md-block font-monospace'>";
            table += "<button class='btn btn-sm btn-success p-1 m-1'   type='button'  onclick='document.getElementById(" + curr_player_id + ").play()'>Play</button>";
            table += "<button class='btn btn-sm btn-secondary p-1 m-1' type='button'  onclick='document.getElementById(" + curr_player_id + ").pause()'>Stop</button>";
            table += "</div>";

            table += "</td>";

            //table += "<td> <audio controls> <source src='" + current_audio + "' type='audio/mpeg'> </audio>" + "</td>";
         }
         table += "</tr>";
      }
      table += "</tbody> </table>";
   }

   resultsDiv.innerHTML = table;
}
