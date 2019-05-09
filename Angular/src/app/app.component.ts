import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css', './main.css']
})
export class AppComponent {
  //private LOGO = require("../assets/site-logs/new_era.png")
  title = 'Angular-Flask';
  serverData: JSON;
  exployeeData: JSON;
  selectedFile: File;
  show: boolean = false;
  score: number = 0.0;
  label: string = '';

  constructor(private httpClient: HttpClient) { }

  sayHi(): void {
    this.httpClient.get('http://52.205.9.75/home').subscribe(data => {
      this.serverData = data as JSON;
      console.log(this.serverData)
    })
  }

  onFileChanged(event) {
    this.selectedFile = event.target.files[0]
  }

  onUpload() {
    const uploadData = new FormData();
    uploadData.append('file', this.selectedFile, this.selectedFile.name);
    this.httpClient.post('http://52.205.9.75/home', uploadData)
      .subscribe(event => {
        console.log(event);
        this.processResults(event);
      });
  }

  processResults(data) {
    this.score = data.score;
    this.label = data.label;
  }
}
