const audio_prefix = "audio";
const datasets = ['speech', 'music', 'urban'];
const techniques = ['signals', 'no_phase', 'griffin_lim', 'dgl_biased', 'dgl_unbiased', 'melgan', 'hifigan', 'uniglow', 'squeezewave', 'univnet'];
const n_audio_files = 5; // Change this up to 20
var curr_player_id = 0;
var current_dataset = 0;

document.write("<h1 class='h1 text-center'>Spectrogram Player</h2>");
document.write("<p class='p text-center'>This web page accompanies the mini-paper 'Comparing Signal Estimation Techniques from Magnitude-Only Spectrograms'.<br>Here, you can listen to the audio excerpts reconstructed by the various techniques.</p>");

for(idx_dataset = 0; idx_dataset < datasets.length; idx_dataset++)
{
   current_dataset = datasets[idx_dataset];
   document.write("<h2 class='h2 text-center'>Comparison for " + current_dataset + "</h2>");

   document.write("<table class='table table-striped table-hover'>  <thead class='font-monospace'> <tr>"); // Generate the table head
   for(idx_technique = 0; idx_technique < techniques.length; idx_technique++)
   {
      document.write("<th scope='col'>" + techniques[idx_technique] + "</th>");
   }
   document.write("</tr> </thead>");

   // Generate the body
   document.write("<tbody class='table-group-divider'>");
   for(idx_audio = 0; idx_audio < n_audio_files; idx_audio++) // For each audio file
   {
      document.write("<tr>");
      for(idx_technique = 0; idx_technique < techniques.length; idx_technique++) // For each technique
      {
         var current_audio = audio_prefix + "/" + current_dataset + "/" + techniques[idx_technique] + "_" + current_dataset + "/" + idx_audio + ".m4a";
         console.log(current_audio);

         document.write("<td>");

         curr_player_id++;
         document.write("<audio id='" + curr_player_id + "'> <source src='" + current_audio + "' type='audio/mpeg'> </audio>");

         document.write("<div class='d-grid gap-2 d-md-block font-monospace'>");
         document.write("<button class='btn btn-sm btn-success p-1 m-1'   type='button'  onclick='document.getElementById(" + curr_player_id + ").play()'>Play</button>");
         document.write("<button class='btn btn-sm btn-secondary p-1 m-1' type='button'  onclick='document.getElementById(" + curr_player_id + ").pause()'>Stop</button>");
         document.write("</div>");

         document.write("</td>");

         //document.write("<td> <audio controls> <source src='" + current_audio + "' type='audio/mpeg'> </audio>" + "</td>");
      }
      document.write("</tr>");
   }
   document.write("</tbody> </table>");
}
